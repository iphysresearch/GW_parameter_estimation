import sys
sys.path.append('..')
import os
from model.utils import ffname
from scipy.signal import tukey
import pycbc.psd
import numpy as np
import json
from gwpy.timeseries import TimeSeries
from pycbc.types.frequencyseries import load_frequencyseries
from pathlib import Path


class Generate_PSD:
    def __init__(self, T=8, T_psd=1024, Tseg=4, roll_off=0.4, fs=4096,
                 delta_f=1.0/8, f_min_psd=20, f_max=4096/4.0, num=10,
                 event='GW150914', address='./data'):
        alpha = 2 * roll_off / Tseg
        self.w = tukey(int(Tseg * fs), alpha)
        self.numT = T * fs
        self.numTseg = Tseg * fs
        self.numTpsd = T_psd * fs
        self.seg_len = int(Tseg * fs)
        self.seg_stride = int(Tseg * fs)
        self.randrange = T_psd-T

        self.f_max = f_max
        self.delta_f = delta_f
        self.f_min_psd = f_min_psd

        self.detectors = None
        self.data_psd = None
        self.t_event = None
        self.psd_dict = {}

        if isinstance(event, list):
            print(f'Init {num} PSDs for multiple events...')
            for e in event:
                self.num = num
                self.get_t_event(Path(address), e)
                self.get_data_psd(Path(address), e)
                self._init_hdfs(Path(address), e)
        elif isinstance(event, str):
            self.get_t_event(Path(address), event)
            self.get_data_psd(Path(address), event)
        else:
            raise
        total_dets_considered = list(
            {key.split('_')[1] for key in self.psd_dict.keys()}
        )

        self.psd_det_filename = {det: [key for key in self.psd_dict.keys() if det in key] for det in total_dets_considered}

    def _init_hdfs(self, address, event):
        os.system(f'rm -rf {address / f"{event}_randpsds"}')
        os.mkdir(f'{address / f"{event}_randpsds"}')
        psd_length = int(self.f_max / self.delta_f) + 1
        for ifo in self.detectors:
            for i in range(self.num):
                self.psd_dict[f'{event}_{ifo}_{i}'] = self.pycbc_psd_from_random(ifo, psd_length, self.delta_f, self.f_min_psd)
                self.psd_dict[f'{event}_{ifo}_{i}'].save(str(address / f"{event}_randpsds" / f'{ifo}_{i}.hdf'))

    def _load_hdfs(self, address, event):
        for i in range(10):
            for ifo in self.detectors:
                self.psd_dict[f'{event}_{ifo}_{i}'] = load_frequencyseries(str(address / f"{event}_randpsds" / f'{ifo}_{i}.hdf'))

    def load_psd_from_random(self, det):
        if det == 'ref':
            det = 'H1'
        return self.psd_dict[self.psd_det_filename[det][np.random.randint(self.num)]]

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
