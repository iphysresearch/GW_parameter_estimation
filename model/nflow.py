from nflows import distributions, flows, transforms, utils
import torch
from torch.nn import functional as F
# import nflows.nn.nets as nn_
from resnet import ResidualNet
from transformer import TransformerResidualNet
from umnn import UMNNCouplingTransform
from priordist import PriorDistribution
import time


def create_model(wfdt, input_dim, num_flow_steps, flowmodel, input_shape, embedding_net=None):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.

    This models the posterior distribution p(x|y).

    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y

    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps

    Returns:
        Flow -- the model
    """
    # TODO
    # distribution = PriorDistribution(wfdt, intput_shape)
    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, flowmodel, input_shape)
    flow = flows.Flow(transform, distribution, embedding_net)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        'input_dim': input_dim,
        'num_flow_steps': num_flow_steps,
        'flowmodel': flowmodel,
    }

    return flow


def create_transform(num_flow_steps,
                     param_dim,
                     flowmodel,
                     input_shape):
    """Build a sequence of NSF transforms, which maps parameters x into the
    base distribution u (noise). Transforms are conditioned on strain data y.

    Note that the forward map is f^{-1}(x, y).

    Each step in the sequence consists of
        * A linear transform of x, which in particular permutes components
        * A NSF transform of x, conditioned on y.
    There is one final linear transform at the end.

    This function was adapted from the uci.py example in
    https://github.com/bayesiains/nsf

    Arguments:
        num_flow_steps {int} -- number of transforms in sequence
        param_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        base_transform_kwargs {dict} -- hyperparameters for NSF step

    Returns:
        Transform -- the constructed transform
    """

    return transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(param_dim),
            create_base_transform(i, param_dim,
                                  flowmodel, input_shape)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(param_dim)
    ])


def create_linear_transform(param_dim):
    """Create the composite linear transform PLU.

    Arguments:
        input_dim {int} -- dimension of the space

    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)
    ])


def get_flowmodel_conditioner(flowmodel, input_shape):
    """generate 'transform_net_create_fn'
    """
    conditioner_type = flowmodel.conditioner.name
    conditioner_kwargs = flowmodel.conditioner
    if conditioner_type == 'TransformerConditioner':
        context_tokens = input_shape[-2]
        context_features = input_shape[-1]
        return (lambda in_features, out_features:
                TransformerResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=conditioner_kwargs.hidden_features,
                    context_tokens=context_tokens,
                    context_features=context_features,
                    num_blocks=conditioner_kwargs.num_blocks,
                    ffn_num_hiddens=conditioner_kwargs.ffn_num_hiddens,
                    num_heads=conditioner_kwargs.num_heads,
                    num_layers=conditioner_kwargs.num_layers,
                    dropout=conditioner_kwargs.dropout,
                    )
                )
    elif conditioner_type == 'ResNetConditioner':
        context_features = input_shape[-1]
        if conditioner_kwargs.activation == 'elu':
            activation_fn = F.elu
        elif conditioner_kwargs.activation == 'relu':
            activation_fn = F.relu
        elif conditioner_kwargs.activation == 'leaky_relu':
            activation_fn = F.leaky_relu
        else:
            activation_fn = F.relu   # Default
            print('Invalid activation function specified. Using ReLU.')
        return (lambda in_features, out_features:
                ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=conditioner_kwargs.hidden_dims,
                    context_features=context_features,
                    num_blocks=conditioner_kwargs.num_blocks,
                    activation=activation_fn,
                    dropout=conditioner_kwargs.dropout,
                    use_batch_norm=conditioner_kwargs.batch_norm
                    )
                )
    else:
        raise ValueError


def get_flowmodel_function(flowmodel):
    """generate flow model and its kwargs
    """
    flowmodel_type = flowmodel.name
    if flowmodel_type == 'RQ-NSF(C)':
        kwargs = dict(
            num_bins=flowmodel.num_bins,
            tails='linear',
            tail_bound=flowmodel.tail_bound,
            apply_unconditional_transform=flowmodel.apply_unconditional_transform
        )
        return transforms.PiecewiseRationalQuadraticCouplingTransform, kwargs
    elif flowmodel_type == 'UMNN':
        kwargs = dict(
            integrand_net_layers=flowmodel.integrand_net_layers,
            cond_size=flowmodel.cond_size,
            nb_steps=flowmodel.nb_steps,
            solver=flowmodel.solver,
        )
        return UMNNCouplingTransform, kwargs
    else:
        raise ValueError


def create_base_transform(i,
                          param_dim,
                          flowmodel,
                          input_shape):
    """Build a base NSF transform of x, conditioned on y.

    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).

    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.

    A coupling flow fixes half the components of x, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.

    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on x_fixed and y. The residual
    network consists of a sequence of two-layer fully-connected blocks.

    Arguments:
        i {int} -- index of transform in sequence
        param_dim {int} -- dimensionality of x

    Keyword Arguments:
        context_dim {int} -- dimensionality of y (default: {None})
        hidden_dim {int} -- number of hidden units per layer (default: {512})
        num_transform_blocks {int} -- number of transform blocks comprising the
                                      transform (default: {2})
        activation {str} -- activation function (default: {'relu'})
        dropout_probability {float} -- probability of dropping out a unit
                                       (default: {0.0})
        batch_norm {bool} -- whether to use batch normalization
                             (default: {False})
        num_bins {int} -- number of bins for the spline (default: {8})
        tail_bound {[type]} -- [description] (default: {1.})
        apply_unconditional_transform {bool} -- whether to apply an
                                                unconditional transform to
                                                fixed components
                                                (default: {False})

        base_transform_type {str} -- type of base transform
                                     ([rq-coupling], rq-autoregressive)

    Returns:
        Transform -- the NSF transform
    """
    model, model_kwargs = get_flowmodel_function(flowmodel)
    return model(
        mask=utils.create_alternating_binary_mask(
            param_dim, even=(i % 2 == 0)),
        transform_net_create_fn=get_flowmodel_conditioner(flowmodel, input_shape),
        **model_kwargs,
    )

    if base_transform_type == 'rq-coupling+resnet':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features:
                                     ResidualNet(
                                        in_features=in_features,
                                        out_features=out_features,
                                        hidden_features=hidden_features,
                                        context_features=context_features,
                                        num_blocks=num_blocks,
                                        activation=activation_fn,
                                        dropout=dropout,
                                        use_batch_norm=batch_norm
                                     )
                                     ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )

    elif base_transform_type == 'rq-coupling+transformer':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features:
                                     TransformerResidualNet(
                                         in_features=in_features,
                                         out_features=out_features,
                                         hidden_features=hidden_features,
                                         context_tokens=context_tokens,
                                         context_features=context_features,
                                         num_blocks=num_blocks,
                                         ffn_num_hiddens=ffn_num_hiddens,
                                         num_heads=num_heads,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                        )
                                     ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )

    elif base_transform_type == 'umnn+transformer':
        return UMNNCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features:
                                     TransformerResidualNet(
                                         in_features=in_features,
                                         out_features=out_features,
                                         hidden_features=hidden_features,
                                         context_tokens=context_tokens,
                                         context_features=context_features,
                                         num_blocks=num_blocks,
                                         ffn_num_hiddens=ffn_num_hiddens,
                                         num_heads=num_heads,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                        )
                                     ),
            integrand_net_layers=integrand_net_layers,
            cond_size=cond_size,
            nb_steps=nb_steps,
            solver=solver,
        )

    elif base_transform_type == 'umnn+resnet':  # TODO
        return UMNNCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features:
                                     ResidualNet(
                                        in_features=in_features,
                                        out_features=out_features,
                                        hidden_features=hidden_features,
                                        context_features=context_features,
                                        num_blocks=num_blocks,
                                        activation=activation_fn,
                                        dropout=dropout,
                                        use_batch_norm=batch_norm
                                     )
                                     ),
            integrand_net_layers=integrand_net_layers,
            cond_size=cond_size,
            nb_steps=nb_steps,
            solver=solver,
        )

    elif base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout,
            use_batch_norm=batch_norm
        )

    else:
        raise ValueError
