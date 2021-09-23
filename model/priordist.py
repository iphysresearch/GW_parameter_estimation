import numpy as np
import pandas as pd
import torch

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class PriorDistribution(Distribution):
    """A Prior from Bilby."""

    def __init__(self, wfdt, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self.wfdt = wfdt

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        params = pd.DataFrame(self.wfdt.transform_inv_params(inputs), columns=self.wfdt.var.target_labels)
        return np.asarray([self.wfdt.wfd.prior[key].ln_prob(params[key]) for key in params]).sum(axis=0)
#         torch.as_tensor()

    def _sample(self, num_samples, context):
        if context is None:
            data = self.wfdt.wfd.prior.sample(num_samples)
            params_block = pd.DataFrame({key: data[key]
                                         for key in data.keys()
                                         if key in self.wfdt.var.target_labels}).values
            return torch.as_tensor(self.wfdt.transform_params(params_block))
#             return torch.randn(num_samples, *self._shape, device=self._log_z.device)
#         else:
#             # The value of the context is ignored, only its size and device are taken into account.
#             context_size = context.shape[0]
#             samples = torch.randn(context_size * num_samples, *self._shape,
#                                   device=context.device)
#             return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context): # TODO
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)