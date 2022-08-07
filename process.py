import os
import argparse

from utils import metadata
from utils import dataloader
import torch
from audioset_tagging_cnn.inference import audio_tagging
from tqdm import tqdm

from indicies import name_indicies
from utils.tagging_validation import tagging_validate

import pandas as pd

checkpoint_path = 'ResNet22_mAP=0.430.pth'

parser = argparse.ArgumentParser(description='Script to process sound files recorded by Audiomoth ')
parser.add_argument('--data_path', default='/Users/nicolas/Desktop/EAVT/example/audio/0001/', type=str, help='Path to wav files')
parser.add_argument('--save_path', default='/Users/nicolas/Desktop/EAVT/example/metadata/', type=str, help='Path to save meta data')
args = parser.parse_args()

csvfile = os.path.join(args.save_path, 'indices.csv')

# get meta data file
df_files = metadata.metadata_generator(args.data_path)

# get data loader
dl = dataloader.get_dataloader_site(args.data_path, df_files, Fmin = 1, Fmax = 10**5, batch_size = 12)

df_site = {'datetime':[], 'name':[], 'start':[], 'clipwise_output' : [], 'embedding' :[], 'sorted_indexes':[]}
for ii in name_indicies: df_site[ii] = []

for batch_idx, (inputs, info) in enumerate(tqdm(dl)):

    # print('tototo')
    with torch.no_grad():
        clipwise_output, labels, sorted_indexes, embedding = audio_tagging(inputs, checkpoint_path , usecuda=False)


    for idx, date_ in enumerate(info['date']):
        df_site['datetime'].append(str(date_)) 
        df_site['name'].append(str(info['name'][idx]))
        df_site['start'].append(float(info['start'][idx]))
        df_site['clipwise_output'].append(clipwise_output[idx])
        df_site['sorted_indexes'].append(sorted_indexes[idx])
        df_site['embedding'].append(embedding[idx])
        for key in info['ecoac'].keys():
            df_site[key].append(float(info['ecoac'][key].numpy()[idx])) 
    

Df_tagging = tagging_validate(df_site)


## Dataframe with only ecoacoustic indices and important metadata 
Df_eco = pd.DataFrame()
Df_eco['name'] = df_site['name']
Df_eco['start'] = df_site['start']
Df_eco['datetime'] = df_site['datetime']
for key in info['ecoac'].keys():
    Df_eco[key] = df_site[key]

## Fusing with the dataframe containing only the ecoacoustic indices 

Df_final = pd.merge(Df_tagging,Df_eco,on=['name','start','datetime'])
Df_final.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)
# Df_tagging.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)