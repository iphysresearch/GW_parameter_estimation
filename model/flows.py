from nflows import distributions, flows, transforms, utils
import torch
from torch.nn import functional as F
# import nflows.nn.nets as nn_
from resnet import ResidualNet
from transformer import TransformerResidualNet
from umnn import UMNNCouplingTransform
import time


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


def create_base_transform(i,
                          param_dim,
                          num_bins=8,                           # rq-coupling
                          tail_bound=1.,                        # rq-coupling
                          apply_unconditional_transform=False,  # rq-coupling

                          base_transform_type='rq-coupling+transformer',
                          # Conditioner:
                          activation='relu',                # resnet
                          batch_norm=False,                 # resnet
                          context_features=2048,            # resnet / transformer
                          hidden_features=32,               # resnet / transformer
                          num_blocks=2,                     # resnet / transformer
                          dropout=0.1,                      # resnet / transformer
                          context_tokens=60,                # transformer
                          ffn_num_hiddens=32,               # transformer
                          num_heads=2,                      # transformer
                          num_layers=2,                     # transformer

                          integrand_net_layers=[50, 50, 50],  # for UMNN
                          cond_size=20,                       # for UMNN
                          nb_steps=20,                        # for UMNN
                          solver="CCParallel",                # for UMNN
                          ):
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

    if activation == 'elu':
        activation_fn = F.elu
    elif activation == 'relu':
        activation_fn = F.relu
    elif activation == 'leaky_relu':
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.relu   # Default
        print('Invalid activation function specified. Using ReLU.')

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


def create_transform(num_flow_steps,
                     param_dim,
                     base_transform_kwargs):
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
                                  **base_transform_kwargs)
        ]) for i in range(num_flow_steps)
    ] + [
        create_linear_transform(param_dim)
    ])


def create_NDE_model(input_dim, num_flow_steps,
                     base_transform_kwargs):
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

    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, base_transform_kwargs)
    flow = flows.Flow(transform, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        'input_dim': input_dim,
        'num_flow_steps': num_flow_steps,
        'base_transform_kwargs': base_transform_kwargs
    }

    return flow


anneal_duration = 50
anneal_max = 3.0


# def anneal_schedule(epoch, quiet=False):
#     if epoch <= anneal_duration:
#         exponent = anneal_max * (anneal_duration - epoch + 1) / anneal_duration
#     else:
#         exponent = 0.0
#     if not quiet:
#         print('Setting annealing exponent to {}.'.format(exponent))
#     return exponent


def train_epoch(flow, train_loader, optimizer, epoch,
                device=None, embedding_net=None,
                output_freq=50, annealing=False):
    """Train model for one epoch.

    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})

    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0

    start_time = time.time()
    for batch_idx, (h, x) in enumerate(train_loader):

        optimizer.zero_grad()

        if device is not None:
            h = h.to(torch.float32).to(device, non_blocking=True)
            x = x.to(torch.float32).to(device, non_blocking=True)

        y = h

        # Compute log prob
        if embedding_net is not None:
            embedding_net.train()
            context = embedding_net(y)
            loss = - flow.log_prob(x, context=context)
        else:
            loss = - flow.log_prob(x, context=y)

        # Keep track of total loss.
        train_loss += loss.sum()

        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tCost: {:.2f}s'.format(
                epoch, batch_idx *
                train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(), time.time()-start_time))
            start_time = time.time()

    train_loss = train_loss.item() / len(train_loader.dataset)
    # train_loss = train_loss.item() / total_weight.item()
    print('Train Epoch: {} \tAverage Loss: {:.4f}'.format(
        epoch, train_loss))

    return train_loss


def test_epoch(flow, test_loader, epoch, device=None, transformer=None,
               annealing=False):
    """Calculate test loss for one epoch.

    Arguments:
        flow {Flow} -- NSF model
        test_loader {DataLoader} -- test set data loader

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPu) (default: {None})

    Returns:
        float -- test loss
    """

    with torch.no_grad():
        flow.eval()

        test_loss = 0.0

        for h, x in test_loader:

            if device is not None:
                h = h.to(torch.float32).to(device, non_blocking=True)
                x = x.to(torch.float32).to(device, non_blocking=True)

            y = h

            # Compute log prob
            if transformer:
                transformer['encoder'].eval()
                loss = - flow.log_prob(x, context=transformer['encoder'](y, transformer['valid_lens']))
            else:
                loss = - flow.log_prob(x, context=y)

            # Keep track of total loss
            test_loss += loss.sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print('Test set: Average loss: {:.4f}\n'
              .format(test_loss))

        return test_loss


def obtain_samples(flow, y, nsamples, device=None, transformer=None, batch_size=512):
    """Draw samples from the posterior.

    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})

    Returns:
        Tensor -- samples
    """

    with torch.no_grad():
        flow.eval()

        y = torch.from_numpy(y).unsqueeze(0).to(device)

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size
        valid_lens = torch.tensor([transformer['valid_lens'][1],]*1).to(device)
        if transformer:
            transformer['encoder'].eval()
            y = transformer['encoder'](y.reshape(-1, 4, transformer['valid_lens'][1]), valid_lens).reshape(-1,4*transformer['valid_lens'][1])
        samples = [flow.sample(batch_size, y) for _ in range(num_batches)]

        
        if num_leftover > 0:
            samples.append(flow.sample(num_leftover, y))

        # The batching in the nsf package seems screwed up, so we had to do it
        # ourselves, as above. They are concatenating on the wrong axis.

        # samples = flow.sample(nsamples, context=y, batch_size=batch_size)

        return torch.cat(samples, dim=1)[0]
