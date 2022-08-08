import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc



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
                        dcc.Dropdown(id='indice 1', options=[]), 
                        html.Br(),html.Br(),
                        dcc.Dropdown(id='indice 2', options=[]),
                        html.Br(),html.Br(),
                        dcc.Dropdown(id='indice 3', options=[]),
                        html.Br(),html.Br(),

                    ]
                ),

            ]
        )

    ], color=color_card, outline=outline
)

