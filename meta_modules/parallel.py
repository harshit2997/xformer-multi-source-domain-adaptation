import torch
import warnings

from torch.nn import DataParallel as DataParallel_
from torch.nn.parallel import DistributedDataParallel
from reid.models.meta_modules.module import MetaModule
from collections import OrderedDict

from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape
from torch.nn.parallel.distributed import _find_tensors


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        if not isinstance(self.module, MetaModule):
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        params = kwargs.pop('params', None)
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        replicas = self._replicate_params(params, inputs_, device_ids,
                                          detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg)
                        for (kwarg, replica) in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            module_params = OrderedDict(self.module.named_parameters())
        else:
            # Temporarily disable the warning if no parameter with key prefix
            # `module` was found. In that case, the original params dictionary
            # is used.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                module_params = self.get_subdict(params, key='module')
            if module_params is None:
                module_params = params

        replicas = _broadcast_coalesced_reshape(list(module_params.values()),
                                                device_ids[:len(inputs)],
                                                detach)
        replicas = tuple(OrderedDict(zip(module_params.keys(), replica))
                         for replica in replicas)
        return replicas


class MetaDistributedDataParallel(DistributedDataParallel, MetaModule):
    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()

        params = kwargs.pop('params', None)
        if params is not None:
            params = self.get_subdict(params, 'module')
            kwargs.__setitem__('params', params)

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

