import torch
from nflows.transforms.base import Transform
from UMNN import NeuralIntegral, ParallelNeuralIntegral
import torch.nn as nn


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.


class IntegrandNet(nn.Module):
    def __init__(self, hidden, cond_in):
        super(IntegrandNet, self).__init__()
        l1 = [1 + cond_in] + hidden
        l2 = hidden + [1]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x, h):
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(nb_batch, -1, in_d).transpose(1, 2).contiguous().view(nb_batch * in_d, -1)
        return self.net(x_he).view(nb_batch, -1)


class MonotonicNormalizer(nn.Module):
    def __init__(self, integrand_net, cond_size, nb_steps=20, solver="CC"):
        super(MonotonicNormalizer, self).__init__()
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps

    def forward(self, x, h, context=None):
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net,
                                             _flatten(self.integrand_net.parameters()),
                                             h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)

    def inverse_transform(self, z, h, context=None):
        # Old inversion by binary search
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)
        for i in range(25):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min
        return (x_max + x_min) / 2


class CouplingTransform(Transform):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None):
        """
        Constructor.

        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier(),
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split, context
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split, context
            )

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class UMNNCouplingTransform(CouplingTransform):
    """An unconstrained monotonic neural networks coupling layer that transforms the variables.

    Reference:
    > A. Wehenkel and G. Louppe, Unconstrained Monotonic Neural Networks, NeurIPS2019.

    ---- Specific arguments ----
        integrand_net_layers: the layers dimension to put in the integrand network.
        cond_size: The embedding size for the conditioning factors.
        nb_steps: The number of integration steps.
        solver: The quadrature algorithm - CC or CCParallel. Both implements Clenshaw-Curtis quadrature with
        Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps) at once,
        it is faster but requires more memory.

    """
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        integrand_net_layers=[50, 50, 50],
        cond_size=20,
        nb_steps=20,
        solver="CCParallel",
        apply_unconditional_transform=False
    ):

        if apply_unconditional_transform:
            unconditional_transform = lambda features: MonotonicNormalizer(integrand_net_layers, 0, nb_steps, solver)
        else:
            unconditional_transform = None
        self.cond_size = cond_size
        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)

    def _transform_dim_multiplier(self):
        return self.cond_size

    def _coupling_transform_forward(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            z, jac = self.transformer(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = jac.log().sum(1)
            return z, log_det_jac
        else:
            B, C, H, W = inputs.shape
            z, jac = self.transformer(
                inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]),
                transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = jac.log().reshape(B, -1).sum(1)
            return z.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac

    def _coupling_transform_inverse(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            x = self.transformer.inverse_transform(
                inputs,
                transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            z, jac = self.transformer(x, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = -jac.log().sum(1)
            return x, log_det_jac
        else:
            B, C, H, W = inputs.shape
            x = self.transformer.inverse_transform(
                inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]),
                transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            z, jac = self.transformer(x, transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = -jac.log().reshape(B, -1).sum(1)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac
