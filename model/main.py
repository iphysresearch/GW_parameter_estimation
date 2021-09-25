# [W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.  # TODO
# http://www.diracprogram.org/doc/release-12/installation/mkl.html
# https://github.com/PaddlePaddle/Paddle/issues/17615
import os
# os.environ['OMP_NUM_THREADS'] = str(1)
# os.environ['MKL_NUM_THREADS'] = str(1)
# os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
try:
    from gwtoolkit.gw import WaveformDataset
    from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
except ModuleNotFoundError as e:
    print(e, "\nLoading from "+'GWToolkit/')
    import sys
    sys.path.insert(0, 'GWToolkit/')
    sys.path.insert(0, 'model/conformer/')
    from gwtoolkit.gw import WaveformDataset
    from gwtoolkit.torch import (WaveformDatasetTorch, Normalize_params, Patching_data, ToTensor)
    #from conformer.encoder import ConformerEncoder

import flows
import nflow
from transformer import TransformerEncoder
from utils import (MultipleOptimizer,
                   MultipleScheduler,
                   ffname,
                   writer_row,
                   js_divergence,
                   kl_divergence,
                   print_dict)

from vggblock import VGGBlock_causal
from cvt import CvT, Transformer, infer_output_dim
from einops.layers.torch import Rearrange
import time
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nflows.utils import get_num_parameters
# from collections import namedtuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


class GlobalAveragePooling(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.GAP = nn.AvgPool2d((input_shape[-2], 1))

    def forward(self, x):
        return self.GAP(x).squeeze(1)


class PosteriorModel(object):
    def __init__(self,
                 model_dir,
                 events_dir,
                 save_model_name,
                 use_cuda):
        super().__init__()
        if model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir = events_dir
        self.save_model_name = save_model_name
        self.train_history = []
        self.test_history = []
        self.epoch_minimum_test_loss = 1
        self.epoch_cache = 1
        self.test_samples = None
        self.embedding_transformer_kwargs = None

        self.flow_net = None
        self.embedding_net = None
        self.loss = None

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
        self.input_shape = None

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
                                  batch_size, num_workers, classification_ornot=None):
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
            rand_transform_data=rand_transform_data,
            classification_ornot=classification_ornot)
        self.wfdt_test = WaveformDatasetTorch(
            self.wfd, num=num,
            start_time=start_time,
            geocent_time=geocent_time,
            target_optimal_snr_tuple=target_optimal_snr_tuple,
            target_labels=target_labels,
            stimulated_whiten_ornot=stimulated_whiten_ornot,
            transform_data=composed_data,
            transform_params=composed_params,
            rand_transform_data=rand_transform_data,
            classification_ornot=classification_ornot)

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
        self.input_shape = self.wfdt_train.data_block.shape[1:]

    def init_embedding_network(self, embedding_net, optim):
        self.embedding_net = embedding_net
        self.embedding_net.to(self.device)

        if optim.pretrain_embedding_dir is not None:
            optim.pretrain_embedding_dir
            checkpoint = torch.load(optim.pretrain_embedding_dir / ffname(optim.pretrain_embedding_dir, f'e*_{self.save_model_name}')[0],
                                    map_location=self.device)
            dd = checkpoint['embedding_net_state_dict']
            for key in list(dd.keys()):
                if 'classifier' in key:
                    del dd[key]
            print('Load pretrained Embedding Network...')
            # Load embedding_net
            self.embedding_net.load_state_dict(checkpoint['embedding_net_state_dict'])

    def init_rearrange(self, pattern, **kwargs):
        arange = Rearrange(pattern, **kwargs)
        print('\tbefore Rearrange:', self.input_shape)
        self.input_shape = infer_output_dim(arange, self.input_shape)
        print('\tafter Rearrange:', self.input_shape)
        return arange

    def init_vggblock(self, **kwargs):
        # (batch_size, num_channel, time_step, patch_size) => VGGBlock
        input_dim = self.input_shape[-1]
        vggblock = VGGBlock_causal(
            in_channels=self.input_shape[0],
            out_channels=64,
            conv_kernel_size=(3, 3),
            pooling_kernel_size=(3, 2),  # (3,2), (2,2)
            num_conv_layers=2,
            input_dim=input_dim,
            conv_stride=(1, 1),
            conv_padding=(2, 0),
            pool_padding=(0, 0, 2, 0),  # (index[-1]_left, index[-1]_right, index[-2]_left, index[-2]_right)
            conv_dilation=1,
            pool_dilation=1,
            layer_norm=False,
        )

        print('\tbefore VGG:', self.input_shape)
        self.input_shape = infer_output_dim(vggblock, self.input_shape)
        print('\tafter VGG:', self.input_shape)
        return vggblock

    def init_cvt(self, **kwargs):
        cvt = CvT(self.input_shape[-2], self.input_shape[-1], self.input_shape[-3], 1000,
                  kernels=[(1, 7), (1, 3), (1, 3)],
                  strides=[(1, 4), (1, 2), (1, 2)])
        print('\tbefore CvT:', self.input_shape)
        self.input_shape = infer_output_dim(cvt, self.input_shape)
        print('\tafter CvT:', self.input_shape)
        return cvt

    def init_conformer(self, **kwargs):
        conformer = ConformerEncoder(input_dim=self.input_shape[-1], device=self.device)
        print('\tbefore Conformer:', self.input_shape)
        self.input_shape = infer_output_dim(conformer, self.input_shape)
        print('\tafter Conformer:', self.input_shape)
        return conformer

    def init_vanilla_transformer(self, **kwargs):
        # Define input data structure of Transformer
        print('\tbefore vanilla Transformer:', self.input_shape)
        norm_shape = self.input_shape
        kwargs.update({
            'norm_shape': norm_shape,
            'key_size': norm_shape[1],
            'query_size': norm_shape[1],
            'value_size': norm_shape[1],
            'num_hiddens': norm_shape[1],
            'ffn_num_input': norm_shape[1],
        })
        self.embedding_transformer_kwargs = kwargs
        print('\tafter vanilla Transformer:', self.input_shape)
        print('\tInit a vanilla Transformer:')
        print_dict(kwargs, 3, '\t\t')
        return TransformerEncoder(**kwargs)

    def init_global_average_pooling(self):
        print('\tbefore Global Average Pooling:', self.input_shape)
        gap = GlobalAveragePooling(self.input_shape)
        self.input_shape = infer_output_dim(gap, self.input_shape)
        print('\tafter Global Average Pooling:', self.input_shape)
        return gap

    def init_classifier(self):
        print('\tbefore Classifier:', self.input_shape)
        classifier = nn.Sequential(
            nn.Linear(
                in_features=self.input_shape[0],
                out_features=64,  # self.representation_size,
                ),
            nn.Tanh(),
            nn.Linear(
                in_features=64,
                out_features=2,  # self.num_classes,
                )
        )
        self.input_shape = infer_output_dim(classifier, self.input_shape)
        print('\tafter Classifier:', self.input_shape)
        return classifier

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
            context_features = self.input_shape[-1]
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
            context_tokens = self.input_shape[-2]
            context_features = self.input_shape[-1]
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

    def init_nflow_network(self, flowmodel, num_flow_steps):
        print_dict({key: flowmodel.__dict__[key]
                    for key in flowmodel.__dict__.keys()
                    if key != 'conditioner'}, ncol=1, prex='\t\t')
        print_dict(flowmodel.conditioner.__dict__, ncol=2, prex='\t\t')

        flow_creator = nflow.create_model
        self.flow_net = flow_creator(wfdt=self.wfdt_train,
                                     input_dim=len(self.target_labels),
                                     num_flow_steps=num_flow_steps,
                                     flowmodel=flowmodel,
                                     input_shape=self.input_shape,
                                     embedding_net=self.embedding_net,
                                     )
        self.flow_net.to(self.device)
        print(f"Num of params:\
              \n=> {get_num_parameters(self.embedding_net):>15,d} | embedding\
              \n=> {get_num_parameters(self.flow_net):>15,d} | nflow")

    def init_training(self, optim):
        if optim.lr_embedding and optim.lr_flow:
            print('Init MultipleOptimizer for flow_net and embedding_net.')
            op1 = torch.optim.Adam(self.flow_net.parameters(), lr=optim.lr_flow)
            op2 = torch.optim.Adam(self.embedding_net.parameters(), lr=optim.lr_embedding)
            self.optimizer = MultipleOptimizer(op1, op2)
        elif optim.lr_flow:
            print('Init optimizer for flow_net only.')
            self.optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=optim.lr_flow)
            self.embedding_net = None  # TODO
        elif optim.lr_embedding:
            print('Init optimizer for embedding_net only.')
            self.optimizer = torch.optim.Adam(self.embedding_net.parameters(), lr=optim.lr_embedding)
            self.loss = nn.CrossEntropyLoss()

        if optim.lr_annealing is True:  # TODO
            if optim.lr_anneal_method == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=optim.steplr_step_size,
                    gamma=optim.steplr_gamma)
            elif optim.lr_embedding and optim.lr_flow:
                print('Init MultipleScheduler (cosine) for flow_net and embedding_net.')
                lr_sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(op1, T_max=optim.total_epochs)
                lr_sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(op2, T_max=optim.total_epochs)
                self.scheduler = MultipleScheduler(lr_sch1, lr_sch2)
            elif optim.lr_anneal_method == 'cosine':
                print('Init scheduler (cosine) for flow_net/embedding_net only.')
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=optim.total_epochs,
                )
            elif optim.lr_anneal_method == 'cosineWR':
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

        start_time = time.time()
        for batch_idx, (h, x) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            if self.device is not None:
                h = h.to(torch.float32).to(self.device, non_blocking=True)
                x = x.to(torch.long if self.loss is not None else torch.float32).to(self.device, non_blocking=True)

            # Compute log prob
            if (self.embedding_net is not None) and (self.flow_net is not None):
                self.embedding_net.train()
                self.flow_net.train()
                loss = - self.flow_net.log_prob(x, context=self.embedding_net(h))
            elif self.flow_net is not None:
                self.flow_net.train()
                loss = - self.flow_net.log_prob(x, context=h)
            elif self.embedding_net is not None:
                self.embedding_net.train()
                loss = self.loss(self.embedding_net(h), x)

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
        print(f'Train Epoch: {epoch} \tAverage Loss: {train_loss:.4f}')

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

            test_loss = 0.0
            for h, x in self.test_loader:

                if self.device is not None:
                    h = h.to(torch.float32).to(self.device, non_blocking=True)
                    x = x.to(torch.long if self.loss is not None else torch.float32).to(self.device, non_blocking=True)


                # Compute log prob
                if (self.embedding_net is not None) and (self.flow_net is not None):
                    self.embedding_net.eval()
                    self.flow_net.eval()
                    loss = - self.flow_net.log_prob(x, context=self.embedding_net(h))
                elif self.flow_net is not None:
                    self.flow_net.eval()
                    loss = - self.flow_net.log_prob(x, context=h)
                elif self.embedding_net is not None:
                    self.embedding_net.eval()
                    loss = self.loss(self.embedding_net(h), x)

                # Keep track of total loss
                test_loss += loss.sum()

            test_loss = test_loss.item() / len(self.test_loader.dataset)
            print(f'Test set: Average loss: {test_loss:.4f}')

            return test_loss

    def train(self, total_epochs, output_freq, inference):
        print('Starting timer')
        start_time = time.time()
        for epoch in range(self.epoch_cache, self.epoch_cache + total_epochs):

            try:
                print(
                    'Learning rate:\n\tflow_net:\t',
                    '\n\tembedding_net:\t '.join(
                        str(Dict['param_groups'][0]['lr'])
                        for Dict in self.optimizer.state_dict()
                    ),
                )
            except TypeError:
                print('Learning rate: {}'.format(
                    self.optimizer.state_dict()['param_groups'][0]['lr']))

            self.train_loader.dataset.update()
            self.test_loader.dataset.update()

            train_loss = self.train_epoch(epoch, output_freq)
            test_loss = self.test_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_history.append(train_loss)
            self.test_history.append(test_loss)

            self.epoch_cache = epoch + 1
            print(self.epoch_minimum_test_loss)
            self._logging_to_file(epoch)
            if self.flow_net is not None:
                # Log/Plot/Save the history to file
                self._plot_kljs_history(epoch, inference.event)
                self._save_kljs_history(epoch, **inference.__dict__)
                if ((output_freq is not None) and (epoch == self.epoch_minimum_test_loss)):
                    self._save_model(epoch)
                    self._save_test_samples()
            elif ((output_freq is not None) and (epoch == self.epoch_minimum_test_loss)):
                self._save_model(epoch)
        print('Stopping timer.')
        stop_time = time.time()
        print(f'Training time (including validation): {stop_time - start_time} seconds')

    @staticmethod
    def _plot_to(ylabel, p, filename):
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(p / filename)
        plt.close()

    def _logging_to_file(self, epoch):
        # Log the history to file

        # Make column headers if this is the first epoch
        if epoch == 1:
            writer_row(self.model_dir, 'loss_history.txt', 'w',
                       [epoch, self.train_history[-1], self.test_history[-1]])
        else:
            writer_row(self.model_dir, 'loss_history.txt', 'a',
                       [epoch, self.train_history[-1], self.test_history[-1]])

            data_history = np.loadtxt(self.model_dir / 'loss_history.txt')
            # Plot
            plt.figure()
            plt.plot(data_history[:, 0],
                     data_history[:, 1], '*--', label='train')
            plt.plot(data_history[:, 0],
                     data_history[:, 2], '*--', label='test')
            self._plot_to('Loss', self.model_dir, 'loss_history.png')
            self.epoch_minimum_test_loss = int(data_history[
                np.argmin(data_history[:, 2]), 0])


    def _save_model(self, epoch):
        for f in ffname(self.model_dir, f'e*_{self.save_model_name}'):
            os.remove(self.model_dir / f)
        print(f'Saving model as e{epoch}_{self.save_model_name}\n')
        self.save_model(filename=f'e{epoch}_{self.save_model_name}')

    def _save_kljs_history(self, epoch, event, nsamples_target_event,
                           flow, fhigh, sample_rate, batch_size,
                           start_time, duration, bilby_dir):

        self.test_samples = self._get_test_samples(event, nsamples_target_event,
                                                   flow, fhigh, sample_rate, batch_size,
                                                   start_time, duration)
        bilby_samples = self._load_a_bilby_samples(event, nsamples_target_event, bilby_dir)

        if epoch == 1:
            writer_row(self.model_dir, 'js_history.txt', 'w', self.target_labels)
            writer_row(self.model_dir, 'kl_history.txt', 'w', self.target_labels)
        writer_row(self.model_dir, 'js_history.txt', 'a',
                   [js_divergence([self.test_samples[:, i],
                                   bilby_samples[:, i]]) for i in range(len(self.target_labels))])
        writer_row(self.model_dir, 'kl_history.txt', 'a',
                   [kl_divergence([self.test_samples[:, i],
                                   bilby_samples[:, i]]) for i in range(len(self.target_labels))])

    def _plot_kljs_history(self, epoch, event):
        if epoch <= 1:
            return
        for s in ['js', 'kl']:
            jsdf = pd.read_csv(self.model_dir / f'{s}_history.txt', sep='\t')
            plt.figure()
            plt.fill_between(range(len(jsdf)),
                             jsdf.mean(axis=1),
                             jsdf.max(axis=1), alpha=0.6, label=event)
            self._plot_to(f'{str.upper(s)} div.', self.model_dir, f'{s}_history.png')

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
                print('No pattern ' + pat + f' found in {p}!')
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

    def save_model(self, filename='model.pt'):
        cache_dict = {  # TODO
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_cache': self.epoch_cache,
            'epoch_minimum_test_loss': self.epoch_minimum_test_loss,
        }
        if self.flow_net is not None:
            cache_dict['flow_net_hyperparams'] = self.flow_net.model_hyperparams
            cache_dict['flow_net_state_dict'] = self.flow_net.state_dict()
        if self.embedding_net is not None:
            cache_dict['embedding_net_state_dict'] = self.embedding_net.state_dict()
            cache_dict['embedding_transformer_kwargs'] = self.embedding_transformer_kwargs
            # cache_dict['embedding_net_attention_weights'] = self.embedding_net.attention_weights  # REVIEW

        if self.scheduler is not None:
            cache_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(cache_dict, self.model_dir / filename)

    def _save_test_samples(self):  # TODO  which event?
        np.save(self.model_dir / 'test_event_samples', self.test_samples)

    def load_model(self):

        checkpoint = torch.load(self.model_dir / ffname(self.model_dir, f'e*_{self.save_model_name}')[0],
                                map_location=self.device)
        flow_net_hyperparams = checkpoint['flow_net_hyperparams']
        assert flow_net_hyperparams['input_dim'] == len(self.target_labels)
        self.base_transform_kwargs = flow_net_hyperparams['base_transform_kwargs']

        if self.embedding_net is not None:
            print('Load Embedding Network...')
            # Load embedding_net
            self.embedding_transformer_kwargs = checkpoint['embedding_transformer_kwargs']
            embedding_net = nn.Sequential(
                self.init_vanilla_transformer(self.embedding_transformer_kwargs),
            )
            self.init_embedding_network(embedding_net)
            self.embedding_net.load_state_dict(checkpoint['embedding_net_state_dict'])

        # Load flow_net
        print('Load Normalizing Flow Network...')
        print('\tNumber of transforms in flow sequence:', flow_net_hyperparams['num_flow_steps'])
        flow_creator = nflow.create_model
        self.flow_net = flow_creator(wfdt=self.wfdt_train,
                                     input_dim=flow_net_hyperparams['input_dim'],
                                     num_flow_steps=flow_net_hyperparams['num_flow_steps'],
                                     flowmodel=flow_net_hyperparams['flowmodel'],
                                     input_shape=self.input_shape,
                                     embedding_net=self.embedding_net,
                                     )
        self.flow_net.to(self.device)
        self.flow_net.load_state_dict(checkpoint['flow_net_state_dict'])

        # Load loss history
        with open(self.model_dir / 'loss_history.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.train_history.append(float(row[1]))
                self.test_history.append(float(row[2]))

        # Set the epoch to the correct value. This is needed to resume
        # training.
        self.epoch_cache = checkpoint['epoch_cache']
        self.epoch_minimum_test_loss = checkpoint['epoch_minimum_test_loss']

        # Store the list of detectors the model was trained with
        # self.detectors = checkpoint['detectors']


def conv_bn_relu_block(X, **kwargs):
    default_kwargs = dict(
        channels=(32, 32, 384),
        kernels=[(3, 7), (3, 5), 1],
        strides=[(1, 2), (2, 2), 1],
        padding=[1, 1, 1],
    )
    default_kwargs.update(kwargs)
    default_kwargs['channels'] = (X.shape[1], ) + default_kwargs['channels']
    channels = default_kwargs['channels']
    kernels = default_kwargs['kernels']
    strides = default_kwargs['strides']
    padding = default_kwargs['padding']
    conv1 = nn.Conv2d if isinstance(kernels[0], tuple) else nn.Conv1d
    bn1 = nn.BatchNorm2d if isinstance(kernels[0], tuple) else nn.BatchNorm1d
    conv2 = nn.Conv2d if isinstance(kernels[1], tuple) else nn.Conv1d
    bn2 = nn.BatchNorm2d if isinstance(kernels[1], tuple) else nn.BatchNorm1d
    conv3 = nn.Conv2d if isinstance(kernels[2], tuple) else nn.Conv1d
    bn3 = nn.BatchNorm2d if isinstance(kernels[2], tuple) else nn.BatchNorm1d
    convblock = nn.Sequential(
        conv1(channels[0], channels[1], kernels[0], strides[0], padding[0]),
        bn1(channels[1]),
        nn.ReLU(inplace=True),
        conv2(channels[1], channels[2], kernels[1], strides[1], padding[1]),
        bn2(channels[2]),
        nn.ReLU(inplace=True),
        nn.Flatten(-2,-1) if isinstance(kernels[1], tuple) else nn.Identity(),
        conv3(channels[2], channels[3], kernels[2], strides[2], padding[2]),
        bn3(channels[3]),
        #nn.Dropout(dropout_value),
    )
    return convblock, convblock(X)


def transfomer_block(X, **kwargs):
    default_kwargs = dict(
        isrel_pos_encoding=False,
        ispso_encoding=False,
        vocab_size=0,  # 0 for embeding only
        ffn_num_hiddens=1536,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        valid_lens=None,
    )
    default_kwargs.update(kwargs)
    norm_shape = [X.shape[-2],
                  X.shape[-1]]
    default_kwargs.update({
        'norm_shape': norm_shape,
        'key_size': norm_shape[1],
        'query_size': norm_shape[1],
        'value_size': norm_shape[1],
        'num_hiddens': norm_shape[1],
        'ffn_num_input': norm_shape[1],
    })

    return TransformerEncoder(**default_kwargs), X


def main():
    args = parse_args()
    print_dict(args.__dict__, ncol=8, prex='\t')
    input()

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
        # ###################################################################################################
        # ###################################################################################################
        # Init pm.wfd and pm.wfd.load_prior_source_detector
        pm.init_WaveformDataset(args.waveform.sampling_frequency, args.waveform.duration,
                                args.waveform.conversion,
                                args.waveform.base, args.waveform.detectors, waveform_arguments,
                                filename=args.prior_dir)
        if args.prior_dir is not None:
            print('Using priors in', args.prior_dir)
        # ###################################################################################################
        # ###################################################################################################
        # Init WaveformDatasetTorch & DataLoader + pm.input_shape/pm.target_labels ##########################
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
            # output: (b c h)
        ])
        rand_transform_data = transforms.Compose([
            ToTensor(),
        ])
        composed_params = transforms.Compose([
            Normalize_params(args.waveform.norm_params_kind,
                             wfd=pm.wfd, labels=target_labels,
                             feature_range=(-1, 1)),
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

        if args.model_source == 'new':
            # ###################################################################################################
            # ###################################################################################################
            # Init embedding network ############################################################################
            print('Init Embedding Network...')
            embedding_transformer_kwargs = dict(
                isrel_pos_encoding=True,
                ispso_encoding=False,
                vocab_size=0,  # 0 for embeding only
                ffn_num_hiddens=args.transformer_embedding.ffn_num_hiddens,
                num_heads=args.transformer_embedding.num_heads,
                num_layers=args.transformer_embedding.num_layers,
                dropout=args.transformer_embedding.dropout,
                valid_lens=None,
            )
            embedding_net = nn.Sequential(
                # pm.init_rearrange('b c h -> b c 1 h'),
                pm.init_rearrange('b (c t) h -> b c t h', c=2),
                pm.init_vggblock(),
                pm.init_rearrange('b c h w -> b (c h) w'),
                pm.init_vanilla_transformer(embedding_transformer_kwargs),
            )
            embedding_net = nn.Sequential(  #TODO
                ConformerEncoder(input_dim=pm.input_shape[-1], device=pm.device),
            )
            embedding_net = nn.Sequential(
                pm.init_rearrange('b c h -> b c 1 h'),
                # pm.init_rearrange('b (c t) h -> b c t h', c=2),
                pm.init_cvt(),
            )
            pm.init_embedding_network(embedding_net)  # pn.embedding_net.to(pm.device)

            # ###################################################################################################
            # ###################################################################################################
            # Init nflow network ################################################################################
            print('Init Normalizing Flow Network...')
            print(f'\tNumber of transforms in flow sequence: {args.num_flow_steps}')
            pm.get_base_transform_kwargs(args)
            pm.init_nflow_network(args.num_flow_steps)
        elif args.model_source == 'existing':
            pm.load_model()

        # #######################################################################################################
        # #######################################################################################################
        # Init training #########################################################################################
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

        try:
            pm.train(args.train.total_epochs, args.train.output_freq, inference_events_kwargs)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print('Finished!')


if __name__ == "__main__":
    main()
