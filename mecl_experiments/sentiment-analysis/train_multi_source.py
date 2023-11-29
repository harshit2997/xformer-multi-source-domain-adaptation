from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np

import torch
from torch.backends import cudnn
from meta_modules.parallel import MetaDistributedDataParallel

from utils.logging import Logger
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR
from utils.distributed_utils import dist_init



def configuration():
    parser = argparse.ArgumentParser(description="train simple person re-identification models")

    # distributed
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--port', type=str, metavar='PATH', default='23446')

    # data
    parser.add_argument("--train_source", type=str, default="data/sentiment-dataset")


    # parser.add_argument('--train-lists', nargs='+', type=str, required=True)
    # parser.add_argument('--validate', action='store_true', help='validation when training')
    # parser.add_argument('--query-list', type=str, default='')
    # parser.add_argument('--gallery-list', type=str, default='')
    # parser.add_argument('--root', type=str, required=True)
    # parser.add_argument('-b', '--batch-size', type=int, default=64)
    # parser.add_argument('-j', '--workers', type=int, default=4)
    # parser.add_argument('--height', type=int, default=256, help="input height")
    # parser.add_argument('--width', type=int, default=128, help="input width")
    # parser.add_argument('--num-instances', type=int, default=4)

    # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50')
    # parser.add_argument('--num_features', type=int, default=0)
    # parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--metric', type=str, default='linear')
    # parser.add_argument('--scale', type=float, default=30.0)
    # parser.add_argument('--metric_margin', type=float, default=0.30)

    # optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW')
    # parser.add_argument('--scheduler', type=str, default='step_lr', choices=['step_lr', 'cosine_lr'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of new parameters, for pretrained ")
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=1000)
    # parser.add_argument('--milestones', nargs='+', type=int, default=[4000, 8000],
    #                     help='milestones for the learning rate decay')
    # training configs
    # parser.add_argument('--resume', type=str, default='', metavar='PATH')
    # parser.add_argument('--iters', type=int, default=12000)
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--print-freq', type=int, default=50)
    # parser.add_argument('--save-freq', type=int, default=2000)
    # parser.add_argument('--refresh-freq', type=int, default=1000)
    # parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # parser.add_argument('--fp16', action='store_true', help="training only")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))

    # alpha scheduler
    parser.add_argument('--alpha-scheduler', type=str, default='constant', choices=['step', 'constant'])
    parser.add_argument('--alpha-milestones', nargs='+', type=int, default=[4000, 8000])
    parser.add_argument('--alpha', type=float, default=0.1)

    # uniform loss
    parser.add_argument('--uniform_weight', type=float, default=0.1)
    parser.add_argument('--unif_t', type=float, default=2.0)
    parser.add_argument('--q_size', type=int, default=16)

    parser.add_argument('--re_weight', type=float, default=0.25)

    args = parser.parse_args()
    return args


class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = args

    @staticmethod
    def build_optimizer(model, args):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise AttributeError('Not support such optimizer `{}`'.format(args.optimizer))

        return optimizer

    @staticmethod
    def build_scheduler(optimizer, args):
        if args.scheduler == 'step_lr':
            lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01,
                                             warmup_iters=args.warmup_step)
        elif args.scheduler == 'cosine_lr':
            lr_scheduler = WarmupCosineLR(optimizer, max_iters=args.iters, warmup_factor=0.01,
                                          warmup_iters=args.warmup_step)
        else:
            raise AttributeError('Not support such scheduler `{}`'.format(args.scheduler))

        return lr_scheduler

    @staticmethod
    def model_factory(args):
        if args.arch == 'resnet_ibn50a':
            return ResNetIBNBase()
        if args.arch == 'resnet50':
            return ResNetBase()

    def model_creator(self, num_classes, args):
        feat = self.model_factory(args)
        cls = Classifier(feat.num_features, num_classes)
        mlp = MLP(feat.num_features, feat.num_features)

        feat.cuda()
        cls.cuda()
        mlp.cuda()

        feat_op = self.build_optimizer(feat, args)
        cls_op = self.build_optimizer(cls, args)
        mlp_op = self.build_optimizer(mlp, args)

        feat = MetaDistributedDataParallel(feat, device_ids=[torch.cuda.current_device()],
                                           output_device=torch.cuda.current_device(), broadcast_buffers=False,
                                           find_unused_parameters=True)

        cls = MetaDistributedDataParallel(cls, device_ids=[torch.cuda.current_device()],
                                          output_device=torch.cuda.current_device(), broadcast_buffers=False,
                                          find_unused_parameters=True)

        mlp = MetaDistributedDataParallel(mlp, device_ids=[torch.cuda.current_device()],
                                          output_device=torch.cuda.current_device(), broadcast_buffers=False,
                                          find_unused_parameters=True)

        feat_sche = self.build_scheduler(feat_op, args)
        cls_sche = self.build_scheduler(cls_op, args)
        mlp_sche = self.build_scheduler(mlp_op, args)

        return (feat, cls, mlp), (feat_op, cls_op, mlp_op), (feat_sche, cls_sche, mlp_sche)

    @staticmethod
    def build_validator(args):
        if not args.validate:
            return None, None, None
        data_builder = DataBuilder(args)
        test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)
        return test_loader, query_dataset.data, gallery_dataset.data

    def run(self):
        args = self.args
        is_distributed = dist_init(args)

        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

        data_loaders = []
        models_list = []
        classifiers = []
        mlps = []
        model_optimizers = []
        classifier_optimizers = []
        mlp_optimizers = []
        model_schedulers = []
        classifier_schedulers = []
        mlp_schedulers = []

        data_builder = DataBuilder(args)
        for train_list in args.train_lists:
            train_loader, train_set = data_builder.build_data(is_train=True, image_list=train_list)
            data_loaders.append(train_loader)

            model, optimizer, scheduler = self.model_creator(train_set.num_classes, args)
            models_list.append(model[0])
            classifiers.append(model[1])
            mlps.append(model[2])
            model_optimizers.append(optimizer[0])
            classifier_optimizers.append(optimizer[1])
            mlp_optimizers.append(optimizer[2])
            model_schedulers.append(scheduler[0])
            classifier_schedulers.append(scheduler[1])
            mlp_schedulers.append(scheduler[2])

        test_loader, query, gallery = self.build_validator(args)

        trainer = MultiSourceTrainer(models_list, classifiers, mlps, model_optimizers,
                                     classifier_optimizers, mlp_optimizers, model_schedulers, classifier_schedulers,
                                     mlp_schedulers, args)

        trainer.train_multi_source(zip(*data_loaders), test_loader, query, gallery)


if __name__ == '__main__':
    cfg = configuration()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    runner = Runner(cfg)
    runner.run()