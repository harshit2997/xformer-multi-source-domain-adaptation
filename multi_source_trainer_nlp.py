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
from utils import accuracy
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy

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

        self.ema_cls = copy.deepcopy(self.classifiers[0])
        self.set_requires_grad(self.ema_cls, False)

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

    def _update_ema_variables_acc(self, ema_model, models, alpha, global_step):

        for ema_param in ema_model.parameters():
            ema_param.data.mul_(alpha)

        for model in models:
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.add_((1 - alpha)/len(models), param.data)            

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
        masks_list = []

        for item in inputs_data:
            input_ids, masks, labels, domains = item
            inputs = input_ids.cuda()
            targets = labels.cuda()
            masks = masks.cuda()
            inputs_list.append(inputs)
            targets_list.append(targets)
            masks_list.append(masks)

        return inputs_list, targets_list, masks_list

    def _get_align_loss(self, feature, target):
        feature = torch.nn.functional.normalize(feature)
        target = torch.nn.functional.normalize(target)
        loss = torch.mean(2 - 2 * torch.sum(feature*target, dim=1))
        return loss

    def run(self, meta_train_index, meta_test_index, inputs_list, targets_list, masks_list, current_iter, target_features):
        # meta train
        feat = self.models[meta_train_index]
        cls = self.classifiers[meta_train_index]
        mlp = self.mlps[meta_train_index]
        inputs_train = inputs_list[meta_train_index]
        targets_train = targets_list[meta_train_index]
        masks_train = masks_list[meta_train_index]

        self.set_requires_grad(cls, True)
        feature, bn_feature = feat(inputs_train, attention_mask = masks_train)

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
        old_weights = self.gradient_update_parameters(feat, loss, step_size=self.get_alpha(current_iter))

        cls_test = self.classifiers[meta_test_index]
        mlp_test = self.mlps[meta_test_index]
        inputs_test = inputs_list[meta_test_index]
        targets_test = targets_list[meta_test_index]
        masks_test = masks_list[meta_test_index]
        self.set_requires_grad(cls_test, False)

        feats_test, bn_feat_test = feat(inputs_test, attention_mask = masks_test)        
        logits_test = cls_test(bn_feat_test)

        self.set_requires_grad(mlp_test, False)
        project_feature_test = mlp_test(bn_feat_test)

        self.restore_model_weights(feat, old_weights)

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

    def train_multi_source(self, data_loaders, validation_evaluator, domain, ind_val_evaluators = None):
        end = time.time()
        best_acc, best_iter, best_f1 = 0, 0, 0

        best_val_acc = None if ind_val_evaluators is None else [0.0 for v in ind_val_evaluators]

        for i, inputs_data in tqdm(enumerate(data_loaders)):
            current_iter = i + 1
            self._refresh_information(current_iter, lr=self.model_schedulers[0].get_lr()[0])
            self.data_time.update(time.time() - end)

            inputs_list, targets_list, masks_list = self._parse_data(inputs_data)

            ema_features = []
            ema_targets = []
            targets_features = []

            with torch.no_grad():
                for idx, inputs in enumerate(inputs_list):
                    targets = targets_list[idx]
                    masks = masks_list[idx]

                    features, bn_features = self.ema_model(inputs, attention_mask = masks)

                    # im_k, idx_unshuffle = self._batch_shuffle_ddp(inputs)
                    # features, bn_features = self.ema_model(im_k)
                    # features = self._batch_unshuffle_ddp(features, idx_unshuffle)
                    # bn_features = self._batch_unshuffle_ddp(bn_features, idx_unshuffle)
                    targets_features.append(bn_features)

                    # tmp_feature_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
                    # tmp_targets_list = [torch.zeros_like(targets) for i in range(dist.get_world_size())]
                    # dist.all_gather(tmp_feature_list, features)
                    # dist.all_gather(tmp_targets_list, targets)
                    # features = torch.cat(tmp_feature_list, dim=0)
                    # targets = torch.cat(tmp_targets_list, dim=0)
                    ema_features.append(features)
                    ema_targets.append(targets)

            self.uniform_loss.deque_enqueue(ema_features, ema_targets)

            for meta_train_index in range(self.num_domains):
                # sample one for meta test
                meta_test_candidates = list(range(self.num_domains))
                meta_test_candidates.remove(meta_train_index)
                random.seed(current_iter*10 + meta_train_index)
                meta_test_index = random.sample(meta_test_candidates, 1)[0]

                loss = self.run(meta_train_index, meta_test_index, inputs_list, targets_list, masks_list, current_iter, targets_features)
                self.model_optimizers[meta_train_index].zero_grad()
                self.classifiers_optimizers[meta_train_index].zero_grad()
                self.mlp_optimizers[meta_train_index].zero_grad()
                # self.classifiers_optimizers[meta_test_index].zero_grad()
                loss.backward()
                self.model_optimizers[meta_train_index].step()
                self.classifiers_optimizers[meta_train_index].step()
                self.mlp_optimizers[meta_train_index].step()
                # self.classifiers_optimizers[meta_test_index].step()

                # self._update_ema_variables(self.ema_cls, self.classifiers[meta_train_index], 0.999, current_iter)                
                # self._update_ema_variables(self.ema_model, self.models[meta_train_index], 0.999, current_iter)

            self._update_ema_variables_acc(self.ema_cls, self.classifiers, 0.99, current_iter)                
            self._update_ema_variables_acc(self.ema_model, self.models, 0.99, current_iter)

            for idx in range(self.num_domains):
                self.model_schedulers[idx].step()
                self.classifiers_schedulers[idx].step()
                self.mlp_schedulers[idx].step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            self._logging(current_iter)

            if current_iter % self.args.save_freq == 0:
                #### FILL EVAL HERE
                (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(self.ema_model, self.ema_cls, return_labels_logits=False)                    
                self.ema_model.train()
                self.set_requires_grad(self.ema_model, False)

                if acc > best_acc:
                    best_acc = acc
                    best_iter = current_iter
                    best_f1 = F1
                    save_checkpoint({'model_state_dict': self.ema_model.state_dict(),
                                    'classifier_state_dict': self.ema_cls.state_dict()},
                                    fpath=osp.join(self.args.model_dir,
                                                   'checkpoints',
                                                   'best_ema_checkpoint_'+domain+'.pth.tar'))

                print('\n Domain {} Iteration {:3d}. Current Acc {:4.1%}, Current F1 {:4.1}\n'
                      .format(domain, current_iter, acc, F1))


                print('\n Domain {} Finished iterations {:3d}. Best iter {:3d}, Best Acc {:4.1%}, F1 Best* {:4.1%}\n'
                      .format(domain, current_iter, best_iter, best_acc, best_f1))

                if ind_val_evaluators is not None and len(ind_val_evaluators) == 3:
                    for val_ind in range(len(ind_val_evaluators)):
                        (_, temp_val_acc, _, _, _), _ = ind_val_evaluators[val_ind].evaluate(
                                                                                    self.models[val_ind],
                                                                                    self.classifiers[val_ind],
                                                                                    return_labels_logits=False)

                        self.models[val_ind].train()
                        self.classifiers[val_ind].train()

                        best_val_acc[val_ind] = max(temp_val_acc, best_val_acc[val_ind])
                        print ("Best val acc at index "+str(val_ind)+" = "+str(best_val_acc[val_ind]))


                end = time.time()

    def _logging(self, cur_iter):
        if not (cur_iter % self.args.print_freq == 0):
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
              .format(cur_iter, self.args.max_iter,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.max_iter - cur_iter) * self.batch_time.avg / 3600,
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
    def gradient_update_parameters(model, loss, step_size=0.001, first_order=False):

        params_gen = model._named_members(lambda module: module._parameters.items())
        params_copy = deepcopy(OrderedDict(params_gen))

        params_gen = model._named_members(lambda module: module._parameters.items())
        params_list = [p for p in list(params_gen) if p[1].requires_grad]
        params = OrderedDict(params_list)

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

        # update model
        params_gen = model._named_members(lambda module: module._parameters.items())

        for (name, param) in params_gen:
            if name in updated_params:
                param.data = updated_params[name].data

        return params_copy
    
    @staticmethod
    def restore_model_weights(model, weights):

        params_gen = model._named_members(lambda module: module._parameters.items())
       
        for (name, param) in params_gen:
            if name in weights:
                param.data = weights[name].data

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