import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn.modules.instancenorm import _InstanceNorm
from reid.models.meta_modules.module import MetaModule


class _MetaInstanceNorm(_InstanceNorm, MetaModule):
    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params.get('weight', None)
        bias = params.get('bias', None)

        return F.instance_norm(input, weight=weight, bias=bias,
                               use_input_stats=self.training or not self.track_running_stats)


class MetaInstanceNorm1d(_MetaInstanceNorm):
    __doc__ = nn.InstanceNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class MetaInstanceNorm2d(_MetaInstanceNorm):
    __doc__ = nn.InstanceNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class MetaInstanceNorm3d(_MetaInstanceNorm):
    __doc__ = nn.InstanceNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
