import sys
sys.path.append('..')
from model.utils import ffname
from scipy.signal import tukey
import pycbc.psd
import numpy as np
import json
from gwpy.timeseries import TimeSeries
from pathlib import Path


class Generate_PSD:
    def __init__(self, T=8, T_psd=1024, Tseg=4, roll_off=0.4, fs=4096, event='GW150914', address='./data'):
        alpha = 2 * roll_off / Tseg
        self.w = tukey(int(Tseg * fs), alpha)
        self.numT = T * fs
        self.numTseg = Tseg * fs
        self.numTpsd = T_psd * fs
        self.seg_len = int(Tseg * fs)
        self.seg_stride = int(Tseg * fs)
        self.randrange = T_psd-T

        self.detectors = None
        self.data_psd = None
        self.t_event = None
        self.get_t_event(Path(address), event)
        self.get_data_psd(Path(address), event)

    def pycbc_psd_from_random(self, det, length, delta_f, low_freq_cutoff):
        if det == 'ref':
            det = 'H1'
        i = np.random.randint(self.randrange)
        data = pycbc.psd.estimate.welch(self.data_psd[det][i:i+self.numT].to_pycbc(),
                                        seg_len=self.seg_len, seg_stride=self.seg_stride, window=self.w,
                                        avg_method='median')
        return pycbc.psd.from_numpy_arrays(data.sample_frequencies.numpy(), data.numpy(), length, delta_f, low_freq_cutoff)

    def get_t_event(self, address, event):
        allevents = {}  # 存放读取的数据
        with open(address / "events" / "GWOSC_allevents_meta.json", 'r', encoding='utf-8') as json_file:
            allevents = json.load(json_file)

        event_version = sorted([key for key, value in allevents.items() if event in key.split('-')[0]],
                               key=lambda x: int(x.split('-')[-1][-1]), reverse=True)[0]
        assert allevents[event_version]['commonName'] == event

        self.detectors = sorted(
            list({meta['detector'] for meta in allevents[event_version]['strain']})
        )

        self.t_event = allevents[event_version]['GPS']  # GPS time of coalescence

    def get_data_psd(self, address, event):
        addr = address / 'events_hdf5' / f'{event}'
        self.data_psd = {}
        for det in self.detectors:
            print(addr / ffname(addr, f'*{det}*hdf5')[0])
            data = TimeSeries.read(addr / ffname(addr, f'*{det}*hdf5')[0], format='hdf5')
            assert (self.t_event > data.times.value[0]) and (self.t_event < data.times.value[-1])

            if (self.t_event > data[self.numTpsd:].times.value[0]) and (self.t_event < data.times.value[-1]):
                self.data_psd[det] = data[:self.numTpsd]
            elif (self.t_event > data.times.value[0]) and (self.t_event < data[-self.numTpsd:].times.value[-1]):
                self.data_psd[det] = data[-self.numTpsd:]
            else:
                raise
            assert (self.t_event > self.data_psd[det].times.value[-1]) or (self.t_event < self.data_psd[det].times.value[0])
