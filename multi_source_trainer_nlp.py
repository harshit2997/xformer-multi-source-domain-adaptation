'''
Derived from https://github.com/Ref-rain/MECL-reid/blob/main/reid/trainer/multi_source_trainer.py
'''

import os.path as osp
import time
from collections import OrderedDict
import random
from bisect import bisect_right
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from evaluation.evaluators import Evaluator
from loss import TripletLoss
from loss.uniformity_loss import Uniformity
from loss.label_smooth import LabelSmoothingCrossEntropy
from meta_modules.module import MetaModule
from utils import accuracy
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint


class MultiSourceTrainer:
    def __init__(self, models, classifiers, mlps, model_optimizers, classifier_optimizers, mlp_optimizers, model_schedulers,
                 classifier_schedulers, mlp_schedulers, args):
        super(MultiSourceTrainer, self).__init__()
        self.args = args

        self.models = models
        self.classifiers = classifiers
        self.mlps = mlps
        self.model_optimizers = model_optimizers
        self.classifiers_optimizers = classifier_optimizers
        self.mlp_optimizers = mlp_optimizers
        self.model_schedulers = model_schedulers
        self.classifiers_schedulers = classifier_schedulers
        self.mlp_schedulers = mlp_schedulers
        self.num_domains = len(models)

        # init_ema_model
        self.ema_model = copy.deepcopy(self.models[0])
        self.set_requires_grad(self.ema_model, False)

        self.ema_cls = copy.deepcopy(self.classifiers)
        for i in range(self.num_domains):
            self.set_requires_grad(self.ema_cls[i], False)

        self.avg_model = copy.deepcopy(self.models[0])
        self.set_requires_grad(self.avg_model, False)

        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()
        self.uniform_loss = Uniformity(q_size=args.q_size, unif_t=args.unif_t).cuda()
        self.label_smooth = LabelSmoothingCrossEntropy().cuda()

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses_inner = AverageMeter()
        self.losses_outer = AverageMeter()
        self.losses_uniform = AverageMeter()
        self.losses_mse = AverageMeter()
        self.precisions_inner = AverageMeter()
        self.precisions_outer = AverageMeter()

    @staticmethod
    def set_requires_grad(model: nn.Module, requires_grad=True):
        for params in model.parameters():
            params.requires_grad = requires_grad

    def get_alpha(self, current_iter=0):
        if self.args.alpha_scheduler == 'constant':
            return self.args.alpha
        if self.args.alpha_scheduler == 'step':
            pos = bisect_right(self.args.alpha_milestones, current_iter-1)
            return self.args.alpha * 0.1 ** pos
        raise AttributeError('Invalid alpha scheduler `{}`'.format(self.args.alpha_scheduler))

    def _update_ema_variables(self, ema_model, model, alpha, global_step):
        # pos = bisect_right(self.args.alpha_milestones, global_step)
        # alpha = 1 - (1-alpha) * 0.1 ** pos
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _update_avg_models(self, avg_model, model_list):
        avg_param_dict = dict()
        for model in model_list:
            for name, param in model.named_parameters():
                if name not in avg_param_dict.keys():
                    avg_param_dict[name] = param.data.clone().detach()
                    continue
                avg_param_dict[name] += param.data.clone().detach()
        for name, params in avg_model.named_parameters():
            params.data = avg_param_dict[name] / len(model_list)

    @staticmethod
    def _parse_data(inputs_data):
        inputs_list = []
        targets_list = []
        for item in inputs_data:
            imgs, _, pids, _, _ = item
            inputs = imgs.cuda()
            targets = pids.cuda()
            inputs_list.append(inputs)
            targets_list.append(targets)

        return inputs_list, targets_list

    def _get_align_loss(self, feature, target):
        feature = torch.nn.functional.normalize(feature)
        target = torch.nn.functional.normalize(target)
        loss = torch.mean(2 - 2 * torch.sum(feature*target, dim=1))
        return loss

    def run(self, meta_train_index, meta_test_index, inputs_list, targets_list, current_iter, target_features):
        # meta train
        feat = self.models[meta_train_index]
        cls = self.classifiers[meta_train_index]
        mlp = self.mlps[meta_train_index]
        inputs_train = inputs_list[meta_train_index]
        targets_train = targets_list[meta_train_index]

        self.set_requires_grad(cls, True)
        feature, bn_feature = feat(inputs_train)

        logits = cls(bn_feature)

        self.set_requires_grad(mlp, True)
        project_feature = mlp(bn_feature)

        loss_ce = self.ce_loss(logits, targets_train)
        loss_tr, _ = self.triplet_loss(feature, targets_train)
        loss_uniform = self.uniform_loss(feature, targets_train, meta_train_index)
        loss_mse = self._get_align_loss(project_feature, target_features[meta_train_index])

        loss = loss_ce + loss_tr

        # meta test
        feat.zero_grad()
        fast_weights_feat = self.gradient_update_parameters(feat, loss, step_size=self.get_alpha(current_iter))

        cls_test = self.classifiers[meta_test_index]
        mlp_test = self.mlps[meta_test_index]
        inputs_test = inputs_list[meta_test_index]
        targets_test = targets_list[meta_test_index]
        self.set_requires_grad(cls_test, False)

        feats_test, bn_feat_test = feat(inputs_test, params=fast_weights_feat)
        logits_test = cls_test(bn_feat_test)

        self.set_requires_grad(mlp_test, False)
        project_feature_test = mlp_test(bn_feat_test)

        loss_ce_test = self.ce_loss(logits_test, targets_test)
        loss_tr_test, _ = self.triplet_loss(feats_test, targets_test)
        loss_mse_test = self._get_align_loss(project_feature_test, target_features[meta_test_index])

        loss_test = loss_ce_test + loss_tr_test

        # log information
        self.losses_inner.update(loss.item())
        self.losses_outer.update(loss_test.item())
        self.losses_mse.update(loss_mse.item())
        self.losses_uniform.update(loss_uniform.item())

        prec_inner, = accuracy(logits.data, targets_train.data)
        prec_outer, = accuracy(logits_test.data, targets_test.data)
        self.precisions_inner.update(prec_inner[0])
        self.precisions_outer.update(prec_outer[0])

        return self.args.re_weight*(loss_mse + loss_mse_test) + self.args.uniform_weight*loss_uniform + (loss_test + loss) / 2

    def train_multi_source(self, data_loaders, val_ds, test_ds):
        end = time.time()
        best_mAP, best_iter = 0, 0

        for i, inputs_data in enumerate(data_loaders):
            current_iter = i + 1
            self._refresh_information(current_iter, lr=self.model_schedulers[0].get_lr()[0])
            self.data_time.update(time.time() - end)

            # inputs_list, targets_list = self._parse_data(inputs_data)

            ema_features = []
            ema_targets = []
            targets_features = []

            with torch.no_grad():
                for idx, inputs in enumerate(inputs_list):
                    targets = targets_list[idx]
                    im_k, idx_unshuffle = self._batch_shuffle_ddp(inputs)
                    features, bn_features = self.ema_model(im_k)
                    features = self._batch_unshuffle_ddp(features, idx_unshuffle)
                    bn_features = self._batch_unshuffle_ddp(bn_features, idx_unshuffle)
                    targets_features.append(bn_features)

                    tmp_feature_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
                    tmp_targets_list = [torch.zeros_like(targets) for i in range(dist.get_world_size())]
                    dist.all_gather(tmp_feature_list, features)
                    dist.all_gather(tmp_targets_list, targets)
                    features = torch.cat(tmp_feature_list, dim=0)
                    targets = torch.cat(tmp_targets_list, dim=0)
                    ema_features.append(features)
                    ema_targets.append(targets)

            self.uniform_loss.deque_enqueue(ema_features, ema_targets)

            for meta_train_index in range(self.num_domains):
                # sample one for meta test
                meta_test_candidates = list(range(self.num_domains))
                meta_test_candidates.remove(meta_train_index)
                random.seed(current_iter*10 + meta_train_index)
                meta_test_index = random.sample(meta_test_candidates, 1)[0]

                loss = self.run(meta_train_index, meta_test_index, inputs_list, targets_list, current_iter, targets_features)
                self.model_optimizers[meta_train_index].zero_grad()
                self.classifiers_optimizers[meta_train_index].zero_grad()
                self.mlp_optimizers[meta_train_index].zero_grad()
                self.classifiers_optimizers[meta_test_index].zero_grad()
                loss.backward()
                self.model_optimizers[meta_train_index].step()
                self.classifiers_optimizers[meta_train_index].step()
                self.mlp_optimizers[meta_train_index].step()
                self.classifiers_optimizers[meta_test_index].step()

                self._update_ema_variables(self.ema_model, self.models[meta_train_index], 0.999, current_iter)

            for idx in range(self.num_domains):
                self.model_schedulers[idx].step()
                self.classifiers_schedulers[idx].step()
                self.mlp_schedulers[idx].step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            self._logging(current_iter)

            if current_iter % self.args.save_freq == 0 and dist.get_rank() == 0:
                # CHANGE EVAL
                if test_loader is not None:  
                    for idx in range(self.num_domains):
                        mAP, rank1 = self._do_valid(self.models[idx], test_loader, query, gallery)
                        if best_mAP < mAP:
                            best_mAP = mAP
                            best_iter = current_iter
                            flag = idx

                    mAP, rank1 = self._do_valid(self.ema_model, test_loader, query, gallery)
                    if best_mAP < mAP:
                        best_mAP = mAP
                        best_iter = current_iter
                        flag = self.num_domains+1

                for idx in range(self.num_domains):
                    save_checkpoint({'state_dict': self.models[idx].state_dict()},
                                    fpath=osp.join(self.args.logs_dir,
                                                   'checkpoints',
                                                   'd{}_checkpoint_{}.pth.tar'.format(idx, current_iter)))

                save_checkpoint({'state_dict': self.ema_model.state_dict()},
                                fpath=osp.join(self.args.logs_dir,
                                               'checkpoints',
                                               'ema_checkpoint_{}.pth.tar'.format(current_iter)))

                print('\n * Finished iterations {:3d}. Best iter {:3d}, Best mAP {:4.1%} In D_{}.\n'
                      .format(current_iter, best_iter, best_mAP, flag))

                end = time.time()

    def _logging(self, cur_iter):
        if not (cur_iter % self.args.print_freq == 0 and dist.get_rank() == 0):
            return
        print('Iter: [{}/{}]\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss_inner {:.3f} ({:.3f})\t'
              'Loss_outer {:.3f} ({:.3f})\t'
              'Loss_neg {:.3f} ({:.3f})\t'
              'Loss_mse {:.3f} ({:.3f})\t'
              'Prec_inner {:.2%} ({:.2%})\t'
              'Prec_outer {:.2%} ({:.2%})'
              .format(cur_iter, self.args.iters,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg,
                      self.losses_inner.val, self.losses_inner.avg,
                      self.losses_outer.val, self.losses_outer.avg,
                      self.losses_uniform.val, self.losses_uniform.avg,
                      self.losses_mse.val, self.losses_mse.avg,
                      self.precisions_inner.val, self.precisions_inner.avg,
                      self.precisions_outer.val, self.precisions_outer.avg))

    def _refresh_information(self, cur_iter, lr):
        if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses_inner = AverageMeter()
            self.losses_outer = AverageMeter()
            self.losses_uniform = AverageMeter()
            self.losses_mse = AverageMeter()
            self.precisions_inner = AverageMeter()
            self.precisions_outer = AverageMeter()
            if dist.get_rank() == 0:
                print("lr = {} \t".format(lr))

    @staticmethod
    def _do_valid(model, test_loader, query, gallery):
        assert query is not None and gallery is not None
        print('=' * 80)
        print("Validating....")
        model.eval()
        evaluator = Evaluator(model)
        mAP, rank1 = evaluator.evaluate(test_loader, query, gallery, verbose=False)
        model.train()
        print('=' * 80)
        return mAP, rank1

    @staticmethod
    def gradient_update_parameters(model, loss, params=None, step_size=0.001, first_order=False):
        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.'
                             'MetaModule`, got `{0}`'.format(type(model)))

        if params is None:
            params = OrderedDict(model.meta_named_parameters())

        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

        updated_params = OrderedDict()

        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                if not param.requires_grad:
                    continue
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                if not param.requires_grad:
                    continue
                updated_params[name] = param - step_size * grad

        return updated_params

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output