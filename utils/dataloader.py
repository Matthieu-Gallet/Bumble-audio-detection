import os
import torch
from torch.utils.data import Dataset

import librosa
import pandas as pd
from tqdm import tqdm
import datetime

import numpy as np
from scipy.signal import resample

from indicies import compute_ecoacoustics
import torchaudio
len_audio_s = 10

NUM_CORE = 4


class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader,  Fmin, Fmax, refdB, savepath):
        self.ref_dB = refdB                
        self.meta = meta_dataloader
        self.Fmin, self.Fmax = Fmin, Fmax
        self.sr_tagging = 32000
        self.savepath = savepath

    def __getitem__(self, idx):

        
        filename = self.meta['filename'][idx]

        wav_o, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)
        wav = torch.tensor(wav_o).view(1, len(wav_o))
        
        torchaudio.save(os.path.join(self.savepath, self.meta['date'][idx].strftime('%Y%m%d_%H%M%S')+ '.flac'), wav, sr, format = 'flac')

        ecoac = compute_ecoacoustics(wav_o, sr, self.ref_dB,self.Fmin, self.Fmax)
        
        wav = resample(wav_o, int(len_audio_s*self.sr_tagging))
        wav = torch.tensor(wav)
        

        return (wav.view(int(len_audio_s*self.sr_tagging)), {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'),
                                                        'ecoac' : ecoac})
        #return([None, None])
    def __len__(self):
        return len(self.meta['filename'])
    
def get_dataloader_site(path_wavfile, meta_site ,Fmin, Fmax, savepath, batch_size=12):

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])

    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        duration = len_file/sr_in
        nb_win = int(duration // len_audio_s )

        # cut into 10 sec segment length
        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            # meta_dataloader = pd.concat([meta_dataloader,pd.DataFrame({'filename': wavfile, 'sr': sr_in, 'start': (
            #     win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta})], ignore_index=True)
            curmeta = pd.DataFrame.from_dict({'filename': [wavfile], 'sr': [sr_in], 'start': (
                    [win*len_audio_s]), 'stop': [((win+1)*len_audio_s)], 'len': [len_file], 'date': meta_site['datetime'][idx] + delta})
            meta_dataloader = pd.concat([meta_dataloader,curmeta], ignore_index=True)

    site_set = Silent_dataset(meta_dataloader.reset_index(drop=True),Fmin, Fmax, np.min(meta_site['dB']), savepath)
    site_set = torch.utils.data.DataLoader(
        site_set,shuffle=False, batch_size=batch_size)#,  num_workers=NUM_CORE)

    return site_set