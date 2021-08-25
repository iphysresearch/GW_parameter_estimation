import argparse
try:
    from gwtoolkit.gw import WaveformDataset
    from model import flows
    from model.transformer import TransformerEncoder    
    from model.utils import MultipleOptimizer, MultipleScheduler
except:
    import sys
    sys.path.insert(0, '../GWToolkit/')
    sys.path.insert(0, '..')
    from gwtoolkit.gw import WaveformDataset
    from model import flows
    from model.transformer import TransformerEncoder
    from model.utils import MultipleOptimizer, MultipleScheduler
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from nflows.utils import get_num_parameters

import os
os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

from collections import namedtuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas
import numpy

class WaveformDatasetTorch(Dataset):
    """Waveform dataset"""

    def __init__(self, wfd, num, start_time, geocent_time, 
                 target_optimal_snr_tuple=None, 
                 target_labels = None,
                 stimulated_whiten_ornot = False,
                 transform_data=None,
                 transform_params=None,
                 rand_transform_data=None):
        """
        Args:

        """
        assert isinstance(num, int)
        assert isinstance(start_time, float)
        assert isinstance(geocent_time, tuple)
        assert (isinstance(target_optimal_snr_tuple, tuple) if None else True)
        Record = namedtuple('Record', 'num start_time geocent_time \
                             target_labels \
                             target_optimal_snr_tuple \
                             stimulated_whiten_ornot')
        self.var = Record(num, start_time, geocent_time, 
                          self._set_target_labels(target_labels),
                          target_optimal_snr_tuple, 
                          stimulated_whiten_ornot)
        self.wfd = wfd
        self.transform_data = transform_data
        self.transform_params = transform_params
        self.rand_transform_data = rand_transform_data

        self.time_array = None
        self.data_block = None
        self.params_block = None
        self._update()

    def __len__(self):
        return self.var.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == 0: # Update self.data_block and self.params_block
            self._update()

        if self.rand_transform_data:
            self.data_block[idx] = self.rand_transform_data(self.data_block[idx])

        return (self.data_block[idx], self.params_block[idx])

    def _update(self):
        # data = 
        # (signal_block,  # Pure signals, (num, len(wfd.dets), ...)
        #  signal_meta,  # parameters of the signals, dict
        #  noise_block,  # Pure colored detector noises, (num, len(wfd.dets), ...)
        #  data_whitened, # mixed signal+noise data whitened by stimulated dets' PSD
        # )
        data = self.wfd.time_waveform_response_block(
            self.var.num,
            self.var.start_time,
            self.var.geocent_time,
            self.var.target_optimal_snr_tuple,
        )
        if self.var.stimulated_whiten_ornot:
            self.data_block = data[3]
        else:
            self.data_block = data[0] + data[2]
        
        # Consider the target params labels
        self.params_block = pandas.DataFrame({key: data[1][key] 
                                              for key in data[1].keys() 
                                              if key in self.var.target_labels}).values

        if self.transform_data:
            self.data_block = self.transform_data(self.data_block)
        if self.transform_params:
            self.params_block = self.transform_params(self.params_block)       

    def _set_target_labels(self, labels=None):
        # default 15 labels
        _labels = ['mass_ratio', 'chirp_mass', 
                   'luminosity_distance', 
                   'dec', 'ra', 'theta_jn', 'psi', 'phase', 
                   'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 
                   'geocent_time']
        return list(set(_labels) & set(labels)) if labels is not None else _labels

#         self.data_block = signal_block + noise_block
#         self.time_array = self.wfd.time_array

class Normalize_params(object):
    """Standardize for parameters
    """

    def __init__(self, kind, **kwargs):
        if kind == 'minmax':
            self.standardize = self.minmaxscaler
            self.standardize_inv = self.minmaxscaler_inverse
            assert 'wfd' in kwargs
            assert 'labels' in kwargs
            self.kwargs = kwargs
            print(f"Standardize by '{kind}' for {len(kwargs['labels'])} parameters.")

    def __call__(self, sample):
        # Self check
        #assert numpy.allclose(samples, self.minmaxscaler_inverse(self.minmaxscaler(sample, wfd, labels), wfd, labels))
        sample = self.standardize(sample, **self.kwargs)
        return sample

    def minmaxscaler(self, X, wfd, labels, feature_range=(0, 1)):
        scale = preprocessing.MinMaxScaler(feature_range=feature_range)
        minimaximum = numpy.asarray([[wfd.prior[label].minimum, wfd.prior[label].maximum] for label in labels])
        scale.fit(minimaximum.T)
        return scale.transform(X)

    def minmaxscaler_inverse(self, X, wfd, labels, feature_range=(0, 1)):
        scale = preprocessing.MinMaxScaler(feature_range=feature_range)
        minimaximum = numpy.asarray([[wfd.prior[label].minimum, self.wfd.prior[label].maximum] for label in labels])
        scale.fit(minimaximum.T)
        return scale.inverse_transform(X)
    
class Patching_data(object):
    """Patching for strain
    """

    def __init__(self, patch_size, overlap, sampling_frequency):
        """
        patch_size, sec
        overlap, sec
        """
        self.nperseg = int(patch_size * sampling_frequency) # sec
        # noverlap must be less than nperseg.
        self.noverlap = int(overlap * self.nperseg)  # [%]
        # nstep = nperseg - noverlap
        print(f'Patching with patch size={patch_size}s and overlap={overlap}s.')

    def __call__(self, x):
        shape = x.shape
        # Created strided array of data segments
        if self.nperseg == 1 and self.noverlap == 0:
            return x[..., numpy.newaxis]
        else:
            # https://stackoverflow.com/a/5568169  also
            # https://iphysresearch.github.io/blog/post/signal_processing/spectral_analysis_scipy/#_fft_helper
            nstep = self.nperseg - self.noverlap
            shape = shape[:-1]+((shape[-1]-self.noverlap)//nstep, self.nperseg)
            strides = x.strides[:-1]+(nstep*x.strides[-1], x.strides[-1])
            return numpy.lib.stride_tricks.as_strided(x, shape=shape,
                                                      strides=strides).reshape(shape[0], -1, self.noverlap)
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)


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

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--data_dir', type=str, required=True)
    dir_parent_parser.add_argument('--basis_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--save_model_name', type=str,
                                   required=True)
    dir_parent_parser.add_argument('--save_aux_filename', type=str,
                                   required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false',
                                   dest='cuda')
    dir_parent_parser.add_argument(
        '--nsample', type=int, default='1000000')

    activation_parent_parser = argparse.ArgumentParser(add_help=None)
    activation_parent_parser.add_argument(
        '--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu')
    train_parent_parser = argparse.ArgumentParser(add_help=None)
    train_parent_parser.add_argument(
        '--batch_size', type=int, default='512')
    train_parent_parser.add_argument('--lr', type=float, default='0.0001')
    train_parent_parser.add_argument('--lr_transformer', type=float, default='0.0001')
    train_parent_parser.add_argument('--lr_anneal_method',
                                     choices=['step', 'cosine', 'cosineWR'],
                                     default='step')
    train_parent_parser.add_argument('--no_lr_annealing', action='store_false',
                                     dest='lr_annealing')
    train_parent_parser.add_argument(
        '--steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument('--steplr_step_size', type=int,
                                     default=80)
    train_parent_parser.add_argument('--flow_lr', type=float)
    train_parent_parser.add_argument('--epochs', type=int, required=True)
    train_parent_parser.add_argument('--transfer_epochs', type=int, default=0)
    train_parent_parser.add_argument(
        '--output_freq', type=int, default='50')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')
    train_parent_parser.add_argument('--no_kl_annealing', action='store_false',
                                     dest='kl_annealing')
    train_parent_parser.add_argument('--detectors', nargs='+')
    train_parent_parser.add_argument('--truncate_basis', type=int)
    train_parent_parser.add_argument('--snr_threshold', type=float)
    train_parent_parser.add_argument('--distance_prior_fn',
                                     choices=['uniform_distance',
                                              'inverse_distance',
                                              'linear_distance',
                                              'inverse_square_distance',
                                              'bayeswave'])
    train_parent_parser.add_argument('--snr_annealing', action='store_true')
    train_parent_parser.add_argument('--distance_prior', type=float,
                                     nargs=2)
    train_parent_parser.add_argument('--bw_dstar', type=float)

    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    train_parser = mode_subparsers.add_parser(
        'train', description=('Train a network.'))

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'))
    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, train_parent_parser])

    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = True

    nde_parser = type_subparsers.add_parser(
        'nde',
        description=('Build and train a flow from the nde package.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser]
    )
    nde_parser.add_argument('--hidden_dims', type=int, required=True)
    nde_parser.add_argument('--nflows', type=int, required=True)
    nde_parser.add_argument('--num_layers_transformer', type=int, required=True)
    nde_parser.add_argument('--batch_norm', action='store_true')
    nde_parser.add_argument('--nbins', type=int, required=True)
    nde_parser.add_argument('--tail_bound', type=float, default=1.0)
    nde_parser.add_argument('--apply_unconditional_transform',
                            action='store_true')
    nde_parser.add_argument('--dropout_probability', type=float, default=0.0)
    nde_parser.add_argument('--dropout_transformer', type=float, default=0.0)
    nde_parser.add_argument('--num_transform_blocks', type=int, default=2)
    nde_parser.add_argument('--num_heads_transformer', type=int, default=2)
    nde_parser.add_argument('--ffn_num_hiddens_transformer', type=int, default=48)
    nde_parser.add_argument('--base_transform_type', type=str,
                            choices=['rq-coupling', 'rq-autoregressive'],
                            default='rq-coupling')

    ns = Nestedspace()

    return parser.parse_args(namespace=ns)


def main():
    args = parse_args()
    print(args)



if __name__ == "__main__":
    main()
