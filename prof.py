import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import h5py
import torch
import corner
import time
import os

from inference.gwpe_main import PosteriorModel
import inference.waveform as wfg

from inference.nde_flows import obtain_samples
from inference.reduced_basis import SVDBasis
#  from gwpy.frequencyseries import FrequencySeries
from inference.transformer import TransformerEncoder
from nflows.utils import get_num_parameters

event_gps_dict = {
     'GW150914': 1126259462.4,
     'GW151012': 1128678900.4,
     'GW151226': 1135136350.6,
     'GW170104': 1167559936.6,
     'GW170608': 1180922494.5,
     'GW170729': 1185389807.3,
     'GW170809': 1186302519.8,
     'GW170814': 1186741861.5,
     'GW170818': 1187058327.1,
     'GW170823': 1187529256.5
}


def kl_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
        kernel = [kde(i, bw_method='scott') for i in samples]
    except np.linalg.LinAlgError:
        return float("nan")

    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )
    factor = 1.0e-5

    a, b = [k(x) for k in kernel]

    for index in range(len(a)):
        a[index] = max(a[index], max(a) * factor)
    for index in range(len(b)):
        b[index] = max(b[index], max(b) * factor)
    a = np.asarray(a)
    b = np.asarray(b)
    return stats.entropy(a, qk=b, base=base)


def js_divergence(samples, kde=stats.gaussian_kde, decimal=5, base=2.0):
    try:
        kernel = [kde(i) for i in samples]
    except np.linalg.LinAlgError:
        return float("nan")

    x = np.linspace(
        np.min([np.min(i) for i in samples]),
        np.max([np.max(i) for i in samples]),
        100
    )

    a, b = [k(x) for k in kernel]
    a = np.asarray(a)
    b = np.asarray(b)

    m = 1. / 2 * (a + b)
    kl_forward = stats.entropy(a, qk=m, base=base)
    kl_backward = stats.entropy(b, qk=m, base=base)
    return np.round(kl_forward / 2. + kl_backward / 2., decimal)


labels = ['$m_1$', '$m_2$', '$\\phi_c$', '$t_c$', '$d_L$', '$a_1$',
          '$a_2$', '$t_1$', '$t_2$', '$\\phi_{12}$',
          '$\\phi_{jl}$', '$\\theta_{JN}$', '$\\psi$', '$\\alpha$',
          '$\\delta$']
######


model_path = 'models/'
model_name = 'GW150914_sample_uniform_100basis_all_uniform_prior'  # an example
# all_models = os.listdir(model_path)
model_dir = os.path.join(model_path, model_name)
save_dir = os.path.join(model_dir, 'all_epoch_test_samples')
try:
    all_epoch_test_samples = np.load(save_dir + '.npy',  allow_pickle=True).tolist()
    print('Load all_epoch_test_samples..')
    print(all_epoch_test_samples.keys())
except:
    all_epoch_test_samples = {}
    print('Init all_epoch_test_samples..')

# try:
save_model_name = [f for f in os.listdir(model_dir) if ('_model.pt' in f) and ('.e' not in f)][0]
save_aux_filename = [f for f in os.listdir(model_dir) if ('_waveforms_supplementary.hdf5' in f) and ('.e' not in f)][0]
epoch = save_model_name.split('_')[0][1:]
print('Try epoch =', epoch)
assert save_model_name[0] == 'e'
assert save_aux_filename[0] == 'e'

all_test_samples = np.empty((50000*10, 15))
event = 'GW150914'
data_dir = 'data/' + event + '_sample_prior_basis/'
pm = PosteriorModel(model_dir=model_dir,
                    data_dir=data_dir,
                    basis_dir=data_dir,
                    sample_extrinsic_only=False,
                    save_aux_filename=save_aux_filename,
                    save_model_name=save_model_name,
                    use_cuda=True)
pm.transformer = False
pm.load_model(pm.save_model_name)
pm.wfd = wfg.WaveformDataset()
pm.wfd.basis = SVDBasis()
pm.wfd.basis.load(directory=pm.basis_dir)
pm.wfd.Nrb = pm.wfd.basis.n

pm.wfd.load_setting(pm.basis_dir, sample_extrinsic_only=False)
pm.init_waveform_supp(pm.save_aux_filename)

t_event = event_gps_dict[event]  # GPS time of coalescence
T = 8.0  # number of seconds to analyze in a segment
T_buffer = 2.0  # buffer time after the event to include

# Load strain data for event
test_data = 'data/events/{}/strain_FD_whitened.hdf5'.format(event)
event_strain = {}
with h5py.File(test_data, 'r') as f:
    event_strain = {det: f[det][:].astype(np.complex64) for det in pm.detectors}

# Project onto reduced basis

d_RB = {}
for ifo, di in event_strain.items():
    h_RB = pm.wfd.basis.fseries_to_basis_coefficients(di)
    d_RB[ifo] = h_RB
_, pm.event_y = pm.wfd.x_y_from_p_h(np.zeros(pm.wfd.nparams), d_RB, add_noise=False)

transformer = False
#transformer = True
if transformer:
    print('using transformer!')
    truncate_basis = 100
    ffn_num_hiddens_transformer = 2048
    num_heads_transformer = 4
    num_layers_transformer = 512
    dropout_transformer = 0.1
    batch_size = 1024

    norm_shape = [4, truncate_basis]
    transformer = {
        'encoder': TransformerEncoder(
            vocab_size=200,  # for embeding only
            key_size=norm_shape[1],
            query_size=norm_shape[1],
            value_size=norm_shape[1],
            num_hiddens=norm_shape[1],
            norm_shape=norm_shape,
            ffn_num_input=norm_shape[1],
            ffn_num_hiddens=ffn_num_hiddens_transformer,
            num_heads=num_heads_transformer,
            num_layers=num_layers_transformer,
            dropout=dropout_transformer,
            noEmbedding=True,
            )
        }
    transformer['encoder'].to(torch.device(pm.device))
    transformer['encoder'].eval()
    transformer['valid_lens'] = torch.tensor([norm_shape[1], ]*batch_size).to(torch.device(pm.device), non_blocking=True)
    print('Transformer:', get_num_parameters(transformer['encoder']))

print('model:', get_num_parameters(pm.model))
input('Continue?')
save_dict = {}
nsamples_target_event = 50000
for nsamples_target_event in [50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000]:
    save_dict[nsamples_target_event] = []
    for _ in range(50):
        print('Number of samples:', nsamples_target_event)
        start_time = time.time()
        x_samples = obtain_samples(pm.model, pm.event_y, nsamples_target_event, pm.device, transformer, 1024)
        print(f'obtain_samples {nsamples_target_event}: {time.time() - start_time:.4f}')
        x_samples = x_samples.cpu()
        # Rescale parameters. The neural network preferred mean zero and variance one. This undoes that scaling.
        test_samples = pm.wfd.post_process_parameters(x_samples.numpy())
        save_dict[nsamples_target_event].append(time.time() - start_time)
#print(save_dict)
np.save('prof_save_dict', save_dict)
#np.save('prof_save_dict_512layertransformer', save_dict)
