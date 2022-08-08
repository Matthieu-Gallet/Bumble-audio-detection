import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

import torchaudio
from scipy import signal

def get_fig_indices(Df, indices, time_axe):

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("<b>{}</b>".format(indices[0]), "<b>{}</b>".format(indices[1]), "<b>{}</b>".format(indices[2])), vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x = time_axe, y = Df[indices[0]], line = dict(color='black'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scattergl(x = time_axe, y = Df[indices[1]], line = dict(color='blue'), opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scattergl(x = time_axe, y = Df[indices[2]], line = dict(color='green'), opacity=0.5), row=3, col=1)
    
   
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, coloraxis_colorbar=dict(
        title="<b>Probabilité</b>",titleside='right',
                        thicknessmode="pixels", thickness=30,
                        lenmode="pixels", len=400,
                        yanchor="bottom", y=0.0,
                        xanchor="right", x=1.1
                        ),yaxis = {'fixedrange': True})
    fig.update_yaxes({'fixedrange': True}, row=2, col=1)
    fig.update_yaxes({'fixedrange': True}, row=3, col=1)
    
    return(fig)



def get_sample_fig(file, path, transpose, mode, save):
    x, sr = torchaudio.load(os.path.join(path, "audio", file + '.flac'), format='flac')
    x_numpy = x[0,:].numpy()
    N = len(x_numpy)
    #### transpose

    #### Save to tmp assets
    torchaudio.save( save, x, sr, format='flac')

    #### Time frequency representation

    f,t, z = signal.stft(x_numpy, fs = sr, nfft=2048, nperseg=2048, noverlap=2048-256)


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Signal</b>", f"<b>{mode}</b>"), vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x = np.arange(N)/sr, y = x_numpy, line = dict(color='black'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Heatmap(x = t, y = f, z = 10*np.log10(np.abs(z)**2/np.max(np.abs(z)**2)), zmin = -40, zmax = 0), row=2, col=1)

    return fig




if __name__ == '__main__':
    from datetime import datetime
    import argparse
    import pandas as pd
    import os


    parser = argparse.ArgumentParser(description='Script to display sound files recorded by Audiomoth ')
    parser.add_argument('--save_path', default='/Users/nicolas/Desktop/EAVT/example/metadata/', type=str, help='Path to save meta data')
    args = parser.parse_args()

    # Load datas
    Df = pd.read_csv(os.path.join(args.save_path, 'indices.csv'))
    indices = Df.columns[3:]
    time_axe = [datetime.strptime(ii,"%Y%m%d_%H%M%S") for ii in Df['datetime']]

    fig = get_fig_indices(Df, ('anthropophony', "geophony", "biophony"), time_axe)
    fig.show()