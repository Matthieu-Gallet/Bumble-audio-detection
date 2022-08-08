import dash
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import ast
import configparser

from requests import options

from datetime import datetime
import argparse

import utils.fig_generation as fig_g


parser = argparse.ArgumentParser(description='Script to display sound files recorded by Audiomoth ')
parser.add_argument('--save_path', default='/Users/nicolas/Desktop/EAVT/example/metadata/', type=str, help='Path to save meta data')
args = parser.parse_args()

# Load datas
Df = pd.read_csv(os.path.join(args.save_path, 'indices.csv'))
indices = Df.columns[3:]
time_axe = [datetime.strptime(ii,"%Y%m%d_%H%M%S") for ii in Df['datetime']]

app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

fig1 = fig_g.get_fig_indices(Df, ('anthropophony', "geophony", "biophony"), time_axe)
fig2 = fig_g.get_sample_fig(Df['datetime'][0], args.save_path, None, None, 'assets/tps1.flac')

### box 1 (choose indices)


color_card ="dark"
outline =True

file_card = dbc.Card(

    [
        dbc.CardHeader("Indices"),
        dbc.CardBody(
            [
                html.Div(
                    [
                        
                        
                        html.Br(),html.Br(),
                        dcc.Dropdown(id='indice 1', options=indices, value='anthropophony'), 
                        html.Br(),html.Br(),
                        dcc.Dropdown(id='indice 2', options=indices, value="geophony"),
                        html.Br(),html.Br(),
                        dcc.Dropdown(id='indice 3', options=indices, value = "biophony"),
                        html.Br(),html.Br(),

                    ]
                ),

            ]
        )

    ], color=color_card, outline=outline
)

### box 2 (audio params)

file_card_audio = dbc.Card(

    [
        dbc.CardHeader("Audio parameters"),
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Br(),html.Br(),
                        dbc.Label("Filter (Hz)", style={'font-weight': 'bold'}),
                        html.Br(),
                        dbc.Label('fmin  '),
                        dcc.Input(id="fmin", type="number",#,label="Fmin",
                                              value=1, style={'width': 100}),
                        dbc.Label('      fmax'),
                        dcc.Input(id='fmax', type="number",#,label="Fmax",
                                              value=10000, style={'width': 100}),
                        html.Br(),html.Br(),

                        dbc.Label("color lim (dB)", style={'font-weight': 'bold'}),
                        html.Br(),
                        dbc.Label('cmin  '),
                        dcc.Input(id="cmin", type="number",#,label="Fmin",
                                              value=-40, style={'width': 100}),
                        dbc.Label('      cmax'),
                        dcc.Input(id='cmax', type="number",#,label="Fmax",
                                              value=0, style={'width': 100}),
                        
                        html.Br(),html.Br(),
                        dbc.Label("Frequency scale", style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='Scale', options=['STFT', 'MFCC', 'CWT'], value='MFCC'), 
                        html.Br(),html.Br(),
                        dbc.Label("Listen audio frequency band", style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='audio_params', options=['1 Hz - 10 kHz', '10 kHz - 20 kHz', '20 kHz - 30 kHz', '30 kHz - 40 kHz', '40 kHz - 50 kHz'], value='1 Hz - 10 kHz'),
                        html.Br(),
                        html.Audio(id="player", src='assets/tps1.flac', controls=True, style={
                        "width": "100%"})

                    ]
                ),

            ]
        )

    ], color=color_card, outline=outline
)


# Customize Layout
app.layout = dbc.Container(
    [
        html.H1("Data analysis tool box"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([file_card, html.Br()], md=3),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Graph(id='indices_fig', figure=fig1),


                        ]),

                    md=9),
            ],
            align="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col([file_card_audio, html.Br()], md=3),
                dbc.Col(
                    html.Div(
                        [
                            dcc.Graph(id='audio', figure=fig2),
                        ]),

                    md=9),
            ],
            align="center",
        ),
        html.Br(),

             ],
            
        fluid=True)



@app.callback(
    Output('indices_fig', 'figure'),
    Input('indice 1', 'value'),
    Input('indice 2', 'value'),
    Input('indice 3', 'value'),    
)
def update_indices(indice1, indice2, indice3):
   return(fig_g.get_fig_indices(Df, (indice1, indice2, indice3), time_axe))

@app.callback(
    Output('audio', 'figure'),
    Output('player', 'src'),
    Input('indices_fig', 'clickData'),  
    Input('player', 'src'),  
    Input('Scale', 'value'),
    Input('fmin', 'value'),
    Input('fmax', 'value'),
    Input('cmin', 'value'),
    Input('cmax', 'value'),
)
def update_signal(clickData, src, mode, fmin, fmax, cmin, cmax):
    if clickData == None:
        return dash.no_update, dash.no_update
    
    if src == 'assets/tps1.flac':
        src = 'assets/tps2.flac'
    else:
        src = 'assets/tps1.flac'

    idx = clickData['points'][0]['pointNumber']
    print(mode)
    
    return(fig_g.get_sample_fig(Df['datetime'][idx], args.save_path, None, mode, src),  src)

if __name__ == '__main__':
    app.run_server(debug=False, port=8054)