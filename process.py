import os
import argparse
from tqdm import tqdm

import pandas as pd
import torch

from utils import metadata
from utils import dataloader

from utils.tagging_validation import tagging_validate
from audioset_tagging_cnn.inference import audio_tagging

from indicies import name_indicies




checkpoint_path = 'ResNet22_mAP=0.430.pth'

parser = argparse.ArgumentParser(description='Script to process sound files recorded by Audiomoth ')
parser.add_argument('--data_path', default='example/audio/0002/', type=str, help='Path to wav files')
parser.add_argument('--save_path', default='example/metadata/', type=str, help='Path to save meta data')
parser.add_argument('--name', default='', type=str, help='name of measurement')
parser.add_argument('--process_tagging', default=1, type=int, help='Process tagging 0 or 1')
parser.add_argument('--process_indices', default=1, type=int, help='Process indices 0 or 1')
parser.add_argument('--audio_format', default='wav', type=str, help='wav or flac')
parser.add_argument('--length_audio_segment', default=10, type=int, help='Length of analyzing window MUST BE LOWER THAN SIGNAL LENGTH')
parser.add_argument('--save_audio_flac', default=1, type=int, help='Saving audio in flac format (needed to run visualization tool)')

parser.add_argument('--Fmin', default=100, type=float, help='Freq min (filter)')
parser.add_argument('--Fmax', default=10**4, type=float, help='Freq max (filter)')
args = parser.parse_args()

PROCESS_TAG = args.process_tagging
print( PROCESS_TAG)
PROCESS_Indices = args.process_indices
AUDIO_FORMAT = args.audio_format
LEN_AUDIO = args.length_audio_segment

if PROCESS_TAG:
    if LEN_AUDIO != 10:
        raise('With tagging, length_audio_segment must be 10')
    

csvfile = os.path.join(args.save_path, f'indices_{args.name}.csv')
audio_savepath = os.path.join(args.save_path, f'audio_{args.name}')
if not os.path.exists(audio_savepath):
    os.makedirs(audio_savepath)

# get meta data file
df_files = metadata.metadata_generator(args.data_path, AUDIO_FORMAT)
if len(df_files) == 0:
    raise('No audio file found')

# get data loader
dl = dataloader.get_dataloader_site(args.data_path, df_files, Fmin = args.Fmin, Fmax = args.Fmax, savepath = audio_savepath, len_audio_s  = LEN_AUDIO, save_audio=args.save_audio_flac, batch_size = 12)
df_site = {'datetime':[], 'name':[], 'start':[]}
if PROCESS_TAG:
    df_site['clipwise_output'] =  []
    df_site['embedding'] = []  
    df_site['sorted_indexes'] = []
if PROCESS_Indices:
    for ii in name_indicies: df_site[ii] = []

for batch_idx, (inputs, info) in enumerate(tqdm(dl)):
    #print(info)
    if PROCESS_TAG:
        with torch.no_grad():
            clipwise_output, labels, sorted_indexes, embedding = audio_tagging(inputs, checkpoint_path , usecuda=False)

    for idx, date_ in enumerate(info['date']):
        df_site['datetime'].append(str(date_)) 
        df_site['name'].append(str(info['name'][idx]))
        df_site['start'].append(float(info['start'][idx]))
        if PROCESS_TAG:
            df_site['clipwise_output'].append(clipwise_output[idx])
            df_site['sorted_indexes'].append(sorted_indexes[idx])
            df_site['embedding'].append(embedding[idx])
        if PROCESS_Indices:
            for key in info['ecoac'].keys():
                df_site[key].append(float(info['ecoac'][key].numpy()[idx])) 
        
if PROCESS_TAG:
    Df_tagging = tagging_validate(df_site)


## Dataframe with only ecoacoustic indices and important metadata 
if PROCESS_Indices:
    Df_eco = pd.DataFrame()
    Df_eco['name'] = df_site['name']
    Df_eco['start'] = df_site['start']
    Df_eco['datetime'] = df_site['datetime']
    for key in info['ecoac'].keys():
        Df_eco[key] = df_site[key]

## Fusing with the dataframe containing only the ecoacoustic indices 
if PROCESS_Indices and PROCESS_TAG:
    Df_final = pd.merge(Df_tagging,Df_eco,on=['name','start','datetime'])
elif PROCESS_TAG:
    Df_final = Df_tagging
else:
    Df_final = pd.DataFrame(df_site)

Df_final.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)
# Df_tagging.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)