from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from torch import nn

from reid import models
from reid.datasets.data_builder import DataBuilder
from reid.evaluation.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, copy_state_dict
from reid.models.multi_source_resnet_ibn import ResNetIBNBase
from reid.models.multi_source_resnet50 import ResNetBase


def model_factory(args):
    if args.arch == 'resnet_ibn50a':
        return ResNetIBNBase()
    if args.arch == 'resnet50':
        return ResNetBase()
def main_worker(args):
    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    data_builder = DataBuilder(args)
    test_loader, query_dataset, gallery_dataset = data_builder.build_data(is_train=False)

    # Create model
    model = model_factory(args.arch)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model)
    print("Test:")
    evaluator.evaluate(test_loader, query_dataset.data, gallery_dataset.data)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--eval', action='store_true')
    # model
    parser.add_argument('-a', '--arch', type=str, required=True)
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=0.3)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    args = parser.parse_args()

    main_worker(args)
