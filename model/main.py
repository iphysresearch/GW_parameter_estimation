import argparse
try:
    from gwtoolkit.gw import WaveformDataset
    from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
    from model import flows
    from model.transformer import TransformerEncoder
    from model.utils import (MultipleOptimizer,
                             MultipleScheduler,
                             ffname,
                             writer_row,
                             js_divergence,
                             kl_divergence,
                             print_dict)
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '../GWToolkit/')
    sys.path.insert(0, '..')
    from gwtoolkit.gw import WaveformDataset
    from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
    from model import flows
    from model.transformer import TransformerEncoder
    from model.utils import (MultipleOptimizer,
                             MultipleScheduler,
                             ffname,
                             writer_row,
                             js_divergence,
                             kl_divergence,
                             print_dict)
import os
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nflows.utils import get_num_parameters

# from collections import namedtuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)


class PosteriorModel(object):
    def __init__(self,
                 model_dir,
                 events_dir,
                 save_model_name,
                 use_cuda):
        super().__init__()
        self.model_dir = model_dir
        self.events_dir = events_dir
        self.save_model_name = save_model_name
        self.train_history = []
        self.test_history = []
        self.epoch_minimum_test_loss = 1

        self.wfd = None
        self.target_labels = None
        self.wfdt_train = None
        self.wfdt_test = None
        self.train_loader = None
        self.test_loader = None
        self.model_type = None
        self.conditioner_type = None
        self.base_transform_kwargs = {}
        self.optimizer = None
        self.scheduler = None

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_WaveformDataset(self, sampling_frequency, duration, conversion,
                             base, dets, waveform_arguments, filename=None):
        self.wfd = WaveformDataset(sampling_frequency=sampling_frequency,
                                   duration=duration,
                                   conversion=conversion)
        self.wfd.load_prior_source_detector(
            filename=filename,
            base=base,
            dets=dets,
            waveform_arguments=waveform_arguments)

    def init_WaveformDatasetTorch(self, num, start_time, geocent_time, target_optimal_snr_tuple,
                                  target_labels, stimulated_whiten_ornot,
                                  composed_data, composed_params, rand_transform_data,
                                  batch_size, num_workers):
        self.target_labels = target_labels

        # pytorch wrappers
        self.wfdt_train = WaveformDatasetTorch(
            self.wfd, num=num,
            start_time=start_time,
            geocent_time=geocent_time,
            target_optimal_snr_tuple=target_optimal_snr_tuple,
            target_labels=target_labels,
            stimulated_whiten_ornot=stimulated_whiten_ornot,
            transform_data=composed_data,
            transform_params=composed_params,
            rand_transform_data=rand_transform_data)
        self.wfdt_test = WaveformDatasetTorch(
            self.wfd, num=num,
            start_time=start_time,
            geocent_time=geocent_time,
            target_optimal_snr_tuple=target_optimal_snr_tuple,
            target_labels=target_labels,
            stimulated_whiten_ornot=stimulated_whiten_ornot,
            transform_data=composed_data,
            transform_params=composed_params,
            rand_transform_data=rand_transform_data)

        # DataLoader objects
        self.train_loader = DataLoader(
            self.wfdt_train, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))
        self.test_loader = DataLoader(
            self.wfdt_test, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))

    def init_embedding_network(self, embedding_net):
        self.embedding_net = embedding_net
        self.embedding_net.to(self.device)

    def init_vanilla_transformer(self, kwargs):
        # Define input data structure of Transformer
        norm_shape = [self.wfdt_train.data_block.shape[-2],
                      self.wfdt_train.data_block.shape[-1]]
        kwargs.update({
            'norm_shape': norm_shape,
            'key_size': norm_shape[1],
            'query_size': norm_shape[1],
            'value_size': norm_shape[1],
            'num_hiddens': norm_shape[1],
            'ffn_num_input': norm_shape[1],
        })
        print('\tInit a vanilla transformer:')
        print_dict(kwargs, 3, '\t\t')
        return TransformerEncoder(**kwargs)

    def get_base_transform_kwargs(self, args):
        self.model_type = args.model_type
        self.conditioner_type = args.conditioner_type
        self.base_transform_kwargs.update(dict(
            base_transform_type=self.model_type+'+'+self.conditioner_type,
        ))

        if self.model_type == 'rq-coupling':
            print('\tRQ-NSF(C) model:')
            self.base_transform_kwargs.update(dict(
                num_bins=args.rq_coupling_model.num_bins,
                tail_bound=args.rq_coupling_model.tail_bound,
                apply_unconditional_transform=args.rq_coupling_model.apply_unconditional_transform,
            ))
            print_dict(vars(args.rq_coupling_model), 1, '\t\t')
        elif self.model_type == 'umnn':
            print('\tUMNN model:')
            self.base_transform_kwargs.update(dict(
                integrand_net_layers=args.umnn_model.integrand_net_layers,
                cond_size=args.umnn_model.cond_size,
                nb_steps=args.umnn_model.nb_steps,
                solver=args.umnn_model.solver,
            ))
            print_dict(vars(args.umnn_model), 1, '\t\t')
        else:
            raise

        if self.conditioner_type == 'resnet':
            print('\tResNet conditioner:')
            context_features = self.wfdt_train.data_block.shape[-1]
            resnet_cond_kwargs = dict(
                hidden_dims=args.resnet_cond.hidden_dims,
                activation=args.resnet_cond.activation,
                context_features=context_features,
                dropout=args.resnet_cond.dropout,
                num_blocks=args.resnet_cond.num_blocks,
                batch_norm=args.resnet_cond.batch_norm,
            )
            self.base_transform_kwargs.update(resnet_cond_kwargs)
            print_dict(resnet_cond_kwargs, 2, '\t\t')
        elif self.conditioner_type == 'transformer':
            print('\tTransformer conditioner:')
            context_tokens = self.wfdt_train.data_block.shape[-2]
            context_features = self.wfdt_train.data_block.shape[-1]
            transformer_cond_kwargs = dict(
                hidden_features=args.transformer_cond.hidden_features,
                context_tokens=context_tokens,
                context_features=context_features,
                num_blocks=args.transformer_cond.num_blocks,
                ffn_num_hiddens=args.transformer_cond.ffn_num_hiddens,
                num_heads=args.transformer_cond.num_heads,
                dropout=args.transformer_cond.dropout,
                num_layers=args.transformer_cond.num_layers,
            )
            self.base_transform_kwargs.update(transformer_cond_kwargs)
            print_dict(transformer_cond_kwargs, 2, '\t\t')
        else:
            raise

    def init_nflow_network(self, num_flow_steps):
        flow_creator = flows.create_NDE_model
        self.flow_net = flow_creator(input_dim=len(self.target_labels),
                                     num_flow_steps=num_flow_steps,
                                     base_transform_kwargs=self.base_transform_kwargs)
        self.flow_net.to(self.device)

    def init_training(self, kwargs):
        if self.embedding_net is not None:
            print('Init MultipleOptimizer for flow_net and embedding_net.')
            op1 = torch.optim.Adam(self.flow_net.parameters(), lr=kwargs['lr_flow'])
            op2 = torch.optim.Adam(self.embedding_net.parameters(), lr=kwargs['lr_embedding'])
            self.optimizer = MultipleOptimizer(op1, op2)
        else:
            self.optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=kwargs['lr_flow'])

        if kwargs['lr_annealing'] is True:  # TODO
            if kwargs['lr_anneal_method'] == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=kwargs['steplr_step_size'],
                    gamma=kwargs['steplr_gamma'])
            elif self.embedding_net is not None:
                print('Init MultipleScheduler (cosine) for flow_net and embedding_net.')
                lr_sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(op1, T_max=kwargs['total_epochs'])
                lr_sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(op2, T_max=kwargs['total_epochs'])
                self.scheduler = MultipleScheduler(lr_sch1, lr_sch2)
            elif kwargs['lr_anneal_method'] == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=kwargs['total_epochs'],
                )
            elif kwargs['lr_anneal_method'] == 'cosineWR':
                self.scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2
                    )
                )

    def train_epoch(self, epoch, output_freq=50):
        """Train model for one epoch.

        Arguments:
            flow_net {Flow} -- NSF model
            train_loader {DataLoader} -- train set data loader
            optimizer {Optimizer} -- model optimizer
            epoch {int} -- epoch number

        Keyword Arguments:
            embedding_net {Embedding network} -- model
            device {torch.device} -- model device (CPU or GPU) (default: {None})
            output_freq {int} -- frequency for printing status (default: {50})

        Returns:
            float -- average train loss over epoch
        """
        train_loss = 0.0
        self.flow_net.train()

        start_time = time.time()
        for batch_idx, (h, x) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            if self.device is not None:
                h = h.to(torch.float32).to(self.device, non_blocking=True)
                x = x.to(torch.float32).to(self.device, non_blocking=True)

            # Compute log prob
            if self.embedding_net is not None:
                self.embedding_net.train()
                loss = - self.flow_net.log_prob(x, context=self.embedding_net(h))
            else:
                loss = - self.flow_net.log_prob(x, context=h)

            # Keep track of total loss.
            train_loss += loss.sum()

            loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tCost: {:.2f}s'.format(
                    epoch, batch_idx *
                    self.train_loader.batch_size, len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(), time.time()-start_time))
                start_time = time.time()

        train_loss = train_loss.item() / len(self.train_loader.dataset)
        print('Train Epoch: {} \tAverage Loss: {:.4f}'.format(
            epoch, train_loss))

        return train_loss

    def test_epoch(self, epoch):
        """Calculate test loss for one epoch.

        Arguments:
            flow_net {Flow} -- NSF model
            test_loader {DataLoader} -- test set data loader

        Keyword Arguments:
            embedding_net {Embedding network} -- model
            device {torch.device} -- model device (CPU or GPu) (default: {None})

        Returns:
            float -- test loss
        """
        with torch.no_grad():
            self.flow_net.eval()

            test_loss = 0.0
            for h, x in self.test_loader:

                if self.device is not None:
                    h = h.to(torch.float32).to(self.device, non_blocking=True)
                    x = x.to(torch.float32).to(self.device, non_blocking=True)

                # Compute log prob
                if self.embedding_net is not None:
                    self.embedding_net.eval()
                    loss = - self.flow_net.log_prob(x, context=self.embedding_net(h))
                else:
                    loss = - self.flow_net.log_prob(x, context=h)

                # Keep track of total loss
                test_loss += loss.sum()

            test_loss = test_loss.item() / len(self.test_loader.dataset)
            print('Test set: Average loss: {:.4f}\n'
                  .format(test_loss))

            return test_loss

    def train(self, total_epochs, output_freq, kwargs):
        epoch = 1

        for epoch in range(epoch, epoch + total_epochs):

            if self.embedding_net is not None:
                print(
                    'Learning rate:\n\tflow_net:\t',
                    '\n\tembedding_net:\t '.join(
                        str(Dict['param_groups'][0]['lr'])
                        for Dict in self.optimizer.state_dict()
                    ),
                )

            else:
                print('Learning rate: {}'.format(
                    self.optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss = self.train_epoch(epoch, output_freq)
            test_loss = self.test_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_history.append(train_loss)
            self.test_history.append(test_loss)

            # Log/Plot the history to file
            self._logging_to_file(epoch, kwargs)

    @staticmethod
    def _plot_to(ylabel, p, filename):
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(p / filename)
        plt.close()

    def _logging_to_file(self, epoch, kwargs):
        # Log the history to file
        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Make column headers if this is the first epoch
        if epoch == 1:
            writer_row(p, 'loss_history.txt', 'w', [epoch, self.train_history[-1], self.test_history[-1]])
        else:
            writer_row(p, 'loss_history.txt', 'a', [epoch, self.train_history[-1], self.test_history[-1]])

            data_history = np.loadtxt(p / 'loss_history.txt')
            # Plot
            plt.figure()
            plt.plot(data_history[:, 0],
                     data_history[:, 1], '*--', label='train')
            plt.plot(data_history[:, 0],
                     data_history[:, 2], '*--', label='test')
            self._plot_to('Loss', p, 'loss_history.png')
            self.epoch_minimum_test_loss = int(data_history[
                np.argmin(data_history[:, 2]), 0])

        self._save_kljs_history(p, epoch, **kwargs)
        self._plot_kljs_history(p, epoch, kwargs['event'])

    def _save_kljs_history(self, p, epoch, event, nsamples_target_event,
                           flow, fhigh, sample_rate, batch_size,
                           start_time, duration, bilby_event_dir):

        test_samples = self._get_test_samples(event, nsamples_target_event,
                                              flow, fhigh, sample_rate, batch_size,
                                              start_time, duration)
        bilby_samples = self._load_a_bilby_samples(event, nsamples_target_event, bilby_event_dir)

        if epoch == 1:
            writer_row(p, 'js_history.txt', 'w', self.target_labels)
            writer_row(p, 'kl_history.txt', 'w', self.target_labels)
        writer_row(p, 'js_history.txt', 'a',
                   [js_divergence([test_samples[:, i],
                                   bilby_samples[:, i]]) for i in range(len(self.target_labels))])
        writer_row(p, 'kl_history.txt', 'a',
                   [kl_divergence([test_samples[:, i],
                                   bilby_samples[:, i]]) for i in range(len(self.target_labels))])

    def _plot_kljs_history(self, p, epoch, event):
        if epoch <= 1:
            return
        for s in ['js', 'kl']:
            jsdf = pd.read_csv(p / f'{s}_history.txt', sep='\t')
            plt.figure()
            plt.fill_between(range(len(jsdf)),
                             jsdf.mean(axis=1),
                             jsdf.max(axis=1), alpha=0.6, label=event)
            self._plot_to(f'{str.upper(s)} div.', p, f'{s}_history.png')

    def _event_data_transform(self, event, events_dir, flow, fhigh, sample_rate, start_time, duration):
        from gwpy.timeseries import TimeSeries
        from gwpy.signal import filter_design

        p = Path(os.path.join(events_dir, f'{event}'))
        bp = filter_design.bandpass(flow, fhigh, sample_rate)

        event_block = np.empty((1, len(self.wfd.dets), self.wfd.num_t))
        for i, (name, det) in enumerate(self.wfd.dets.items()):
            pat = f'*{name}*.hdf5'
            try:
                strain = TimeSeries.read(p / ffname(p, pat)[0], format='hdf5')
            except IndexError:
                print(f'No pattern '+pat+f' found in {p}!')
                raise
            strain_target_segment = (start_time <= strain.times.value) & (strain.times.value <= start_time+duration)

            event_block[0, i] = det.frequency_to_time_domain(
                det.whiten(det.time_to_frequency_domain(strain.filter(bp, filtfilt=True)[strain_target_segment])[0])
            )[0]
        return self.wfdt_train.transform_data_block(event_block)

    def _obtain_samples(self, data_block, nsamples, batch_size=512):
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
            self.flow_net.eval()

            h = data_block.to(torch.float32).to(self.device, non_blocking=True)

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size
            if self.embedding_net is not None:
                self.embedding_net.eval()
                h = self.embedding_net(h)
            samples = [self.flow_net.sample(batch_size, h) for _ in range(num_batches)]

            if num_leftover > 0:
                samples.append(self.flow_net.sample(num_leftover, h))

        # The batching in the nsf package seems screwed up, so we had to do it
        # ourselves, as above. They are concatenating on the wrong axis.

        # samples = flow.sample(nsamples, context=y, batch_size=batch_size)
        return torch.cat(samples, dim=1)[0]

    def _get_test_samples(self, event, nsamples_target_event,
                          flow, fhigh, sample_rate, batch_size,
                          start_time, duration):
        event_block_transformed = self._event_data_transform(event, self.events_dir,
                                                             flow, fhigh, sample_rate,
                                                             start_time, duration)
        test_samples = self._obtain_samples(event_block_transformed,
                                            nsamples_target_event, batch_size)
        # Rescale parameters.
        # The neural network preferred minmax-style [0, 1] .
        # This undoes that scaling.
        return self.wfdt_train.transform_inv_params(test_samples.cpu())

    def _load_a_bilby_samples(self, event, nsample, bilby_event_dir):
        # event_gps_dict = {
        #     'GW150914': 1126259462.3999023,#1126259462.4,
        #     'GW151012': 1128678900.4,
        #     'GW151226': 1135136350.6,
        #     'GW170104': 1167559936.6,
        #     'GW170608': 1180922494.5,
        #     'GW170729': 1185389807.3,
        #     'GW170809': 1186302519.8,
        #     'GW170817': 1187008882.4,
        #     'GW170814': 1186741861.5,
        #     'GW170818': 1187058327.1,
        #     'GW170823': 1187529256.5
        # }
        event = event.split('_')[0]
        # Load bilby samples
        df = pd.read_csv(os.path.join(bilby_event_dir, '{}_downsampled_posterior_samples.dat'
                                      .format(event)), sep=' ')
        # Shift the time of coalescence by the trigger time
        # bilby_samples[:, 3] = bilby_samples[:, 3] - event_gps_dict[event]
        return df.dropna()[self.target_labels].sample(nsample).values.astype('float64')


class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Model the gravitational-wave parameter '
                     'posterior distribution with neural networks.'))

    # dir
    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--events_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--prior_dir', type=str, default=None)
    dir_parent_parser.add_argument('--save_model_name', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false', dest='cuda')

    # waveform: WaveformDataset
    waveform_parent_parser = argparse.ArgumentParser(add_help=None)
    waveform_parent_parser.add_argument(
        '--waveform.sampling_frequency', type=int, default='4096', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.duration', type=int, default='8', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.conversion',
        choices=['BBH', 'BNS'],
        default='BBH',
        required=True)
    waveform_parent_parser.add_argument(
        '--waveform.waveform_approximant',
        choices=['IMRPhenomPv2', 'SEOBNRv4P'],  # TODO
        default='IMRPhenomPv2',
        required=True)
    waveform_parent_parser.add_argument(
        '--waveform.reference_frequency', type=float, default='50', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.minimum_frequency', type=float, default='20', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.base',
        choices=['bilby', 'pycbc'],  # TODO
        default='bilby',
        required=True)
    waveform_parent_parser.add_argument(
        '--waveform.detectors',
        nargs='+',
        required=True)
    # waveform: WaveformDatasetTorch
    waveform_parent_parser.add_argument(
        '--waveform.target_time', type=float, default='1126259462.3999023', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.buffer_time', type=float, default='2')
    waveform_parent_parser.add_argument(
        '--waveform.patch_size', type=float, default='0.5', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.overlap', type=float, default='0.5', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.stimulated_whiten', action='store_true')
    waveform_parent_parser.add_argument(
        '--waveform.norm_params_kind', type=str, default='minmax', required=True)
    waveform_parent_parser.add_argument(
        '--waveform.target_optimal_snr', nargs='+', type=float, default=(0, 0.))

    # train
    train_parent_parser = argparse.ArgumentParser(add_help=None)
    train_parent_parser.add_argument(
        '--train.epoch_size', type=int, default='512')
    train_parent_parser.add_argument(
        '--train.batch_size', type=int, default='512')
    train_parent_parser.add_argument(
        '--train.num_workers', type=int, default='16')
    train_parent_parser.add_argument(
        '--train.total_epochs', type=int, default='10')
    train_parent_parser.add_argument(
        '--train.lr_flow', type=float, default='0.0001')
    train_parent_parser.add_argument(
        '--train.lr_embedding', type=float, default='0.0001')
    train_parent_parser.add_argument(
        '--train.lr_anneal_method', choices=['step', 'cosine', 'cosineWR'], default='step')
    train_parent_parser.add_argument(
        '--train.no_lr_annealing', action='store_false', dest='train.lr_annealing')
    train_parent_parser.add_argument(
        '--train.steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument(
        '--train.steplr_step_size', type=int, default=80)
    train_parent_parser.add_argument(
        '--train.output_freq', type=int, default='50')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')

    # inference events
    events_parent_parser = argparse.ArgumentParser(add_help=None)
    events_parent_parser.add_argument(
        '--events.batch_size', type=int, default='4', required=True)
    events_parent_parser.add_argument(
        '--events.nsamples_target_event', type=int, default='100', required=True)
    events_parent_parser.add_argument(
        '--events.event', type=str, default='GW150914', required=True)
    events_parent_parser.add_argument(
        '--events.flow', type=float, default='50', required=True)
    events_parent_parser.add_argument(
        '--events.fhigh', type=float, default='250', required=True)
    events_parent_parser.add_argument(
        '--events.sample_rate', type=int, default='4096')
    events_parent_parser.add_argument(
        '--events.start_time', type=float, default='1126259456.3999023', required=True)
    events_parent_parser.add_argument(
        '--events.duration', type=float, default='8', required=True)
    events_parent_parser.add_argument(
        '--events.bilby_dir', type=str, default=None)

    # Embedding network
    embedding_parent_parser = argparse.ArgumentParser(add_help=None)
    embedding_parent_parser.add_argument(
        '--transformer_embedding.ffn_num_hiddens', type=int, default='128')
    embedding_parent_parser.add_argument(
        '--transformer_embedding.num_heads', type=int, default='2')
    embedding_parent_parser.add_argument(
        '--transformer_embedding.num_layers', type=int, default='6')
    embedding_parent_parser.add_argument(
        '--transformer_embedding.dropout', type=float, default='0.1')

    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    # 1 ##      [train]/inference
    train_parser = mode_subparsers.add_parser(
        'train', description=('Train a network.'))

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    # 2.1 ##    [train] - [new]/existing
    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'),
        parents=[dir_parent_parser, waveform_parent_parser, train_parent_parser, events_parent_parser])
    train_new_parser.add_argument(
        '--num_flow_steps', type=int, required=True)

    # 2.2 ##    [train] - new/[existing]
    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, waveform_parent_parser, train_parent_parser, events_parent_parser])

    # 2.1.(1) (coupling function) [train] - [new]/existing
    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = True

    # (coupling function) [train] - [new]/existing - [rq-coupling]/umnn
    coupling_parser_dict = {
        'rq-coupling': type_subparsers.add_parser(
            'rq-coupling',
            description=('Build and train a flow using re-coupling.'),
        )
    }
    coupling_parser_dict['rq-coupling'].add_argument(
        '--rq_coupling_model.num_bins', type=int, required=True)
    coupling_parser_dict['rq-coupling'].add_argument(
        '--rq_coupling_model.tail_bound', type=float, default=1.0)
    coupling_parser_dict['rq-coupling'].add_argument(
        '--rq_coupling_model.apply_unconditional_transform', action='store_true')
    # rq-coupling \
    # --rq_coupling_model.num_bins 8 \

    # (coupling function) [train] - [new]/existing - rq-coupling/[umnn]
    coupling_parser_dict['umnn'] = type_subparsers.add_parser(
        'umnn',
        description=('Build and train a flow using UMNN.'),
    )
    coupling_parser_dict['umnn'].add_argument(
        '--umnn_model.integrand_net_layers', type=int, nargs='+', default=[50, 50, 50])
    coupling_parser_dict['umnn'].add_argument(
        '--umnn_model.cond_size', type=float, default=20)
    coupling_parser_dict['umnn'].add_argument(
        '--umnn_model.nb_steps', type=float, default=20)
    coupling_parser_dict['umnn'].add_argument(
        '--umnn_model.solver', type=str, default="CCParallel")
    # umnn \
    # --umnn_model.integrand_net_layers 50 50 50 \
    # --umnn_model.cond_size 20 \
    # --umnn_model.nb_steps 20 \
    # --umnn_model.solver CCParallel \

    # 2.1.(2) (conditioner function)  [train] - [new]/existing - (coupling function)
    conditioner_parser_dict = {}
    for coupling_name, coupling_parser in coupling_parser_dict.items():
        conditioner_parser_dict[coupling_name] = {}
        conditioner_subparsers = coupling_parser.add_subparsers(dest='conditioner_type')
        conditioner_subparsers.required = True

        # (conditioner function) [train] - [new]/existing - [rq-coupling]/[umnn] - [resnet]/transformer
        conditioner_parser_dict[coupling_name] = conditioner_subparsers.add_parser(
            'resnet',
            description=('Build and train a flow with ResNet conditioner.'),
        )
        conditioner_parser_dict[coupling_name].add_argument(
            '--resnet_cond.hidden_dims', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--resnet_cond.activation',
            choices=['relu', 'leaky_relu', 'elu'],
            default='relu')
        conditioner_parser_dict[coupling_name].add_argument(
            '--resnet_cond.dropout', type=float, default=0.0)
        conditioner_parser_dict[coupling_name].add_argument(
            '--resnet_cond.num_blocks', type=int, default=2)
        conditioner_parser_dict[coupling_name].add_argument(
            '--resnet_cond.batch_norm', action='store_true')
    # resnet \
    # --resnet_cond.hidden_dims 512 \
    # --resnet_cond.activation elu \
    # --resnet_cond.dropout 0.1 \
    # --resnet_cond.num_blocks 10 \
    # --resnet_cond.batch_norm

        # (conditioner function) [train] - [new]/existing - [rq-coupling]/[umnn] - resnet/[transformer]
        conditioner_parser_dict[coupling_name] = conditioner_subparsers.add_parser(
            'transformer',
            description=('Build and train a flow with vanilla Transformer conditioner.'),
            parents=[embedding_parent_parser],
        )
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.hidden_features', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.num_blocks', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.ffn_num_hiddens', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.num_heads', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.num_layers', type=int, required=True)
        conditioner_parser_dict[coupling_name].add_argument(
            '--transformer_cond.dropout', type=float, required=True)
    # transformer \
    # --transformer_cond.hidden_features 4 \
    # --transformer_cond.num_blocks 2 \
    # --transformer_cond.ffn_num_hiddens 16 \
    # --transformer_cond.num_heads 2 \
    # --transformer_cond.num_layers 2 \
    # --transformer_cond.dropout 0.1

    ns = Nestedspace()
    return parser.parse_args(namespace=ns)


def main():
    args = parse_args()
    print(args)
    if args.mode == 'train':
        print('Events directory\t', args.events_dir)
        print('Model directory\t\t', args.model_dir)
        pm = PosteriorModel(model_dir=args.model_dir,
                            events_dir=args.events_dir,
                            save_model_name=args.save_model_name,
                            use_cuda=args.cuda)
        print(f'Save the model as\t{args.save_model_name}')
        print('Device\t\t\t', pm.device)

        print('Init Waveform Dataset...')
        print_dict(vars(args.waveform), 5, '\t')
        waveform_arguments = dict(
            waveform_approximant=args.waveform.waveform_approximant,
            reference_frequency=args.waveform.reference_frequency,
            minimum_frequency=args.waveform.minimum_frequency)
        pm.init_WaveformDataset(args.waveform.sampling_frequency, args.waveform.duration,
                                args.waveform.conversion,
                                args.waveform.base, args.waveform.detectors, waveform_arguments,
                                filename=args.prior_dir)
        if args.prior_dir is not None:
            print('Using priors in', args.prior_dir)
        # Init WaveformDatasetTorch ##################################################$$
        # Hyper parameters start #############################
        print('Init Waveform PyTorch Dataset...')
        target_labels = ['mass_ratio', 'chirp_mass',
                         'luminosity_distance',
                         'dec', 'ra', 'theta_jn', 'psi', 'phase',
                         'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                         'geocent_time']
        start_time = args.waveform.target_time-(args.waveform.duration - args.waveform.buffer_time)
        geocent_time = (args.waveform.target_time-0.1, args.waveform.target_time+0.1)
        # Hyper parameters end ###############################
        composed_data = transforms.Compose([
            Patching_data(patch_size=args.waveform.patch_size,
                          overlap=args.waveform.overlap,
                          sampling_frequency=args.waveform.sampling_frequency),
        ])
        rand_transform_data = transforms.Compose([
            ToTensor(),
        ])
        composed_params = transforms.Compose([
            Normalize_params(args.waveform.norm_params_kind, wfd=pm.wfd, labels=target_labels),
            ToTensor(),
        ])
        pm.init_WaveformDatasetTorch(
            args.train.epoch_size,
            start_time,
            geocent_time,
            tuple(
                float(value) if i else int(value)
                for i, value in enumerate(args.waveform.target_optimal_snr)
            ),
            target_labels,
            args.waveform.stimulated_whiten,
            composed_data,
            composed_params,
            rand_transform_data,
            args.train.batch_size,
            args.train.num_workers,
        )

        # Init embedding network #######################################################
        print('Init Embedding Network...')
        embedding_transformer_kwargs = dict(
            noEmbedding=True,
            vocab_size=200,  # for embeding only
            ffn_num_hiddens=args.transformer_embedding.ffn_num_hiddens,
            num_heads=args.transformer_embedding.num_heads,
            num_layers=args.transformer_embedding.num_layers,
            dropout=args.transformer_embedding.dropout,
            valid_lens=None,
        )
        embedding_net = nn.Sequential(
            pm.init_vanilla_transformer(embedding_transformer_kwargs),
        )
        pm.init_embedding_network(embedding_net)

        # Init nflow network ##########################################################
        print('Init Normalizing Flow Network...')
        print(f'\tNumber of transforms in flow sequence: {args.num_flow_steps}')
        pm.get_base_transform_kwargs(args)
        pm.init_nflow_network(args.num_flow_steps)

        # Init training ###############################################################
        optimization_kwargs = dict(
            total_epochs=args.train.total_epochs,
            lr_flow=args.train.lr_flow,
            lr_embedding=args.train.lr_embedding,
            lr_annealing=args.train.lr_annealing,
            lr_anneal_method=args.train.lr_anneal_method,
            steplr_step_size=args.train.steplr_step_size,
            steplr_gamma=args.train.steplr_gamma,
        )
        pm.init_training(optimization_kwargs)
        print('\tArgumentations for training:')
        print_dict(vars(args.train), 3, '\t\t')

        print(f"Num of params:\
              \n=> {get_num_parameters(pm.embedding_net):>15,d} | embeding\
              \n=> {get_num_parameters(pm.flow_net):>15,d} | nflow")

        # Training ###################################################################
        inference_events_kwargs = dict(
            event=args.events.event,
            nsamples_target_event=args.events.nsamples_target_event,
            flow=args.events.flow,
            fhigh=args.events.fhigh,
            sample_rate=args.events.sample_rate,
            batch_size=args.events.batch_size,
            start_time=args.events.start_time,
            duration=args.events.duration,
            bilby_event_dir=args.events.bilby_dir,
        )
        print('\tInference events during training:')
        print_dict(vars(args.events), 3, '\t\t')
        pm.train(args.train.total_epochs, args.train.output_freq, inference_events_kwargs)


if __name__ == "__main__":
    main()
