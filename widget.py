import dash
import dash_html_components as html
import pandas as pd
import dash_core_components as dcc
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, datetime
import dash
from dash.dependencies import Input, Output
import re
import numpy as np
import yfinance as yf
import pandas as pds
import plotly.graph_objs as go


import plotly.graph_objs as go
o = pds.read_excel("Yahoo_Stock_List.xlsx")
m = o.values
p = [{"label":f"{m[i][1]}", "value":f"{m[i][0]}", "sector":f"{m[i][2]}"} for i in range(len(o))]
select = [{"label":f"{dictionnaire['label']}", "value":f"{dictionnaire['value']}"} for dictionnaire in p]
values = [ f"{dictionnaire['value']}" for dictionnaire in p]
sector = pds.DataFrame([{"value":f"{dictionnaire['value']}", "sector":f"{dictionnaire['sector']}"} for dictionnaire in p])
class widget:
    
    # Header
    affichage = html.Div([dcc.Markdown(" **PORTFOLIO OPTIMIZATION PROJECT** ", style={"font-size":"35px"}),
                         dcc.Markdown(" ** Group N°** ", style={"font-size":"15px"}),
                         dcc.Markdown(" **Lucas Inglese | Margaux Flaus | Florent Fischer** ", style={"font-size":"15px"})],
                        style={"background-color":"#506F85",
                               "margin":"-60px -60px 35px -60px",
                              "color":"white",
                              "height":"250px",
                              "padding":"100px 0px 0px 0px",
                              "textAlign":"center"})
    
    # Strat date dropdown
    start_date = html.Div([dcc.Markdown(" **START** "),

                           dcc.DatePickerSingle(id="Start",
                             min_date_allowed=date(1995, 8, 5),
                             max_date_allowed=datetime.now(),
                             initial_visible_month=date(2017, 8, 5),
                             date=date(2000, 1, 1))], style={"textAlign":"center", "padding":"0px px 0px 15px"})

    # End date dropdown
    end_date = html.Div([dcc.Markdown(" **END** "),

                           dcc.DatePickerSingle(id="End",
                             min_date_allowed=date(1995, 8, 5),
                             max_date_allowed=datetime.now(),
                             initial_visible_month=date(2000, 1, 1),
                             date=date(2021, 1, 1))],style={"textAlign":"center"})

    # Choosing the assets
    selection_des_actifs = html.Div([
                    dcc.Dropdown(id="actifs",
                     options=select,
                    value=["GOOG","TSLA", "" "EURUSD=X","EURGBP=X", "AAL.L"],
                     multi=True)], style={"margin-top":"35px", "width":"50%"})


    # Choosing the optimizeur
    optimizor = html.Div([dcc.Markdown("**OPTIMISATION METHOD CHOICE**"),
                      dcc.RadioItems(id="Optimizor",options=[{'label': "Max sharpe optimisation", 'value': "sharpe"},
                                              {'label': "Max Sortino optimisation", 'value': "sortino"},
                                              {'label': "Mean variance optimisation", 'value': "EV"},
                                              {'label': "Mean variance skweness kurtosis optimisation", 'value': "SK"},
                                              {'label': "Min variance optimisation", 'value': "MV"}],value="sharpe")
                     ], style={})

    # Choosing the benchmarc
    bench = html.Div([dcc.Markdown("**BENCHMARC**"),
                      dcc.RadioItems(id="Bench",options=[{'label': "SP500", 'value': "^GSPC"},
                                              {'label': "CAC40", 'value': "^FCHI"},
                                              {'label': "DJI30", 'value': "^DJI"},
                                              {'label': "DAX30", 'value': "^GDAXI"},
                                              {'label': "EUROSTOC50", 'value': "^STOXX50E"},
                                              {'label': "NQ100", 'value': "^NDX"}],value="^NDX")
                     ], style={})



    # Choosing allow short
    short = html.Div([dcc.Markdown("**SHORT SELLING ALLOWED**"),
                      dcc.RadioItems(id="Short",options=[{'label': "Yes", 'value': True},
                                              {'label': "No", 'value': False}],value=True)
                     ], style={})

    # Choose the period
    dates =  html.Div([start_date, end_date], style={"columnCount": 2})

    # Efficiency border chart
    frontiere = html.Div([dcc.Graph(id="frontiere")], style={"border":"1px rgb(235,235,235)","box-shadow": "1px 1px 1px 1px rgb(195,195,195)",
                                                            "margin-top":"15px"})
    # Cumulative return train
    train = html.Div([dcc.Graph(id="CRTrain")], style={"border":"1px rgb(235,235,235)",
                                           "box-shadow": "1px 1px 1px 1px rgb(195,195,195)"})
    # Cumulative return test
    test = html.Div([dcc.Graph(id="CRTest")], style={"border":"1px rgb(235,235,235)",
                                           "box-shadow": "1px 1px 1px 1px rgb(195,195,195)"})
    # Previsions returns chart
    previsions = html.Div([dcc.Graph(id="CRPrevision")], style={"border":"1px rgb(235,235,235)",
                                           "box-shadow": "1px 1px 1px 1px rgb(195,195,195)"})
    # Chart for drawdon
    drawdown_graph = dcc.Graph(id="drawdown")
    
    # Monthly return heatmap
    tableau = dcc.Graph(id="monthly return heatmap")
    
    # Monthly return omparaison
    bar = dcc.Graph(id="monthly return comparaison")
    
    
    # Div cumulative return
    b = html.Div([train, test, previsions],style={})
    
    # Portfolio Weiht barplot
    w = html.Center(children=[],id="weight", style={"display": "center"})
    
    # Params for statistics
    params = {"font-size":"20px",
             "margin-left":"5px",
              "margin-top":"15px",
             "background-color":"white",
             "border-radius": "5px",
              "height": "75px"
             }

    
    Sharpe = html.Div([html.Div([dcc.Markdown("", id="Sharpe")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Sharpe")], style={"margin-left":"5px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })

    Sortino = html.Div([html.Div([dcc.Markdown("", id="Sortino")],style={"font-size":params["font-size"],
                                                                         "margin-left":params["margin-left"],
                                                                        "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Sortino")], style={"margin-left":"5px"})],
                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-left":"5px",
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })

    drawdown = html.Div([html.Div([dcc.Markdown("", id="max_drawdown")],style={"font-size":params["font-size"],
                                                                               "margin-left":params["margin-left"],
                                                                              "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Drawdown")], style={"margin-left":"5px"})],
                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-left":"5px",
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })

    beta = html.Div([html.Div([dcc.Markdown("", id="beta")],style={"font-size":params["font-size"],
                                                                   "margin-left":params["margin-left"],
                                                                  "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Beta")], style={"margin-left":"5px"})],
                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-left":"5px",
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })

    alpha = html.Div([html.Div([dcc.Markdown("", id="alpha")],style={"font-size":params["font-size"],
                                                                     "margin-left":params["margin-left"],
                                                                    "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Alpha")], style={"margin-left":"5px"})],
                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-left":"5px",

                          "border-radius": params["border-radius"],
                            })

    Stats = html.Div([Sharpe,
                      Sortino,
                      drawdown,
                      beta,
                      alpha], style={"columnCount": 5, "textAlign":"center", "color":"#1F3452"})

    inputs = html.Div([optimizor,
                       bench,
                       short,
                      html.Div([dcc.Markdown("")],style={"height":"73px"})], style={
                                                                                    "background":"white",
                                                                                    "padding":"30px",
                                            "border":"1px rgb(235,235,235)","box-shadow": "1px 1px 1px 1px rgb(195,195,195)"})

    h = html.Div([start_date,
                       end_date,
                  selection_des_actifs
                       ], style={"display":"flex", "padding":"15px",
                                "border-radius": "15px",
                                "border":"1px rgb(235,235,235)","box-shadow": "1px 1px 1px 1px rgb(195,195,195)", "background":"rgb(247,247,247)"})
    
    
    b1d = html.Div([Stats, frontiere],style={})
    bandeau1 = html.Div([inputs, html.Div([dcc.Markdown("")],style={"height":"10px"}),
                         b1d], style={"columnCount": 2, "height":"595px", "margin":"35px 0px 0px 0px"})

    bandeau2 = html.Div([train, test], style={"columnCount": 2,
                                                                              "margin":"35px 0px 0px 0px"
                                                                              } )

    bandeau3 = html.Div([drawdown_graph, previsions,], style={"columnCount": 2,
                                                                                          "margin":"35px 0px 0px 0px"})
    bandeau3_b = html.Div([tableau, dcc.Graph(id="weights")], style={"columnCount": 2,
                                                                                          "margin":"35px 0px 0px 0px"})

    comparaison = html.Div([bar], style={"margin":"35px 0px 0px 0px"})
    
    dashboard = html.Div([affichage,
                           h,
                           bandeau1,
                          bandeau2,
                          bandeau3,
                           bandeau3_b,
                          comparaison],style={"padding": "60px", "background":"rgb(235,235,235)", "margin":"-15px"})


