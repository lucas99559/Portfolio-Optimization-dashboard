external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout =     widget.dashboard
@app.callback(Output("frontiere", "figure"),
              Output("CRTrain", "figure"),
              Output("CRTest", "figure"),
              Output("drawdown", "figure"),
              Output("CRPrevision", "figure"),
              Output("monthly return heatmap", "figure"),
              Output("monthly return comparaison", "figure"),
              Output("weights", "figure"),
              Output("Sharpe","children"),
              Output("Sortino","children"),
              Output("max_drawdown","children"),
              Output("beta","children"),
              Output("alpha","children"),

              
    Input("Start", "date"),
    Input("End", "date"),
    Input("actifs", "value"),
    Input("Optimizor", "value"),
    Input("Bench", "value"),
    Input("Short", "value")
    )


def renvoie(start,end,actifs,Optimizor,ben,short):
    P = PORTFOLIO(start,end,actifs, ben,short)
    
    if Optimizor=="sharpe":
        P.max_sharpe_optimisation()
    elif Optimizor=="sortino":
        P.max_sortino_optimisation()
    elif Optimizor=="EV":
        P.mean_variance_optimisation()
    elif Optimizor=="MV":
        P.min_variance_optimisation()
    elif Optimizor=="SK":
        P.mean_variance_skweness_kurtosis_optimisation()
    else:
        pass
    drawdown = P.drawdown()
    P.statistics()
    frontiere = P.chart_efficiency_borner()
    train = P.pie_chart_sector()
    test = P.chart_cumulative_returns(train=False)
    previsions = P.previsions()
    tableau = P.heatmap_monthly_return()
    comparaison = P.yearly_return_comparaison()
    table = P.bar_plot_weights()
    alpha = f"{np.round(P.alpha*100,1)}%"
    beta = f"{np.round(P.beta,3)}"
    sharpe = f"{np.round(P.sharpe,3)}"
    sortino = f"{np.round(P.sortino,3)}"
    mdrawdown = f"{np.round(P.max_drawdown*100,1)}%"
    print(short)
    
    return frontiere,train,test, drawdown, previsions, tableau, comparaison, table, sharpe, sortino, mdrawdown, beta, alpha
if __name__ == '__main__':
    app.run_server(debug=False,port=8000, host="127.0.0.1")
    