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
o = pds.read_excel("Yahoo_Stock_List.xlsx")
m = o.values
p = [{"label":f"{m[i][1]}", "value":f"{m[i][0]}", "sector":f"{m[i][2]}"} for i in range(len(o))]
select = [{"label":f"{dictionnaire['label']}", "value":f"{dictionnaire['value']}"} for dictionnaire in p]
values = [ f"{dictionnaire['value']}" for dictionnaire in p]
sector = pds.DataFrame([{"value":f"{dictionnaire['value']}", "sector":f"{dictionnaire['sector']}"} for dictionnaire in p])

class PORTFOLIO:
    """ 
        ---------------------------------------------------------------------------------
        | The PORTFOLIO class allow us to do all computation on background to all plots |
        ---------------------------------------------------------------------------------
        |                                                                               |
        | Inputs : - Start (str): Start date to do the optimization                     |
        |                                                                               |
        |          - End (str): End date to do the optimization                         |
        |                                                                               |
        |          - Actifs (list): List containing all the assets that you want        |
        |                                                                               |
        |          - Ben (str): The benchmark that you want to you to the comparaison   |
        |                                                                               |
        |          - Short_selling_allowed (Boolean): True if you want allow short      |
        |            selling False otherwise                                            |
        |                                                                               |
        ---------------------------------------------------------------------------------
        |                                                                               |
        | Methodes: - Max_sharpe_optimisation(): Optimize the portfolio using           |
        |               maximal sharpe ratio                                            |
        |                                                                               |
        |           - Mean_variance_optimisation(): Optimize the portfolio using        |
        |               mean variance criterion                                         |
        |                                                                               |
        |           - Mean_variance_skweness_kurtosis_optimisation(): Optimize the      |
        |               portfolio using mean variance skweness kurtosis criterion       |
        |                                                                               |
        |           - Max_sortino_optimisation(): Optimize the portfolio using          |
        |               maximal sortino ratio                                           |
        |                                                                               |
        |           - Min_variance_optimisation(): Optimize the portfolio using         |
        |               minimal variance criterion                                      |
        |                                                                               |
        |           - Statistics(): Give us all the statistics on the test set and      |
        |               some statistics of the train set to do the chart                |
        |                                                                               |
        |           - Max_sharpe_optimisation(): Optimize the portfolio using           |
        |               maximal sortino ratio                                           |
        |                                                                               |
        |           - Chart_cumulative_returns(train=True): Return the generator of the |
        |               cumulative returns chart                                        |
        |                                                                               |
        |           - Previsions(): Return the generator of the previsions (with 500    |
        |               simulations)                                                    |
        |                                                                               |
        |           - Chart_efficiency_borner(): Efficiency border of the portfolio     |
        |               (on train set)                                                  |
        |                                                                               |
        |           - Drawdown(): Return the generator of the portfolio drawdown        |
        |                                                                               |
        |           - Heatmap_monthly_return((): Return the generator of a heatmap of   |
        |               portfolio monthly return                                        |
        |                                                                               |
        |           - Yearly_return_comparaison(): Return the generator of a barplot of |
        |               portfolio monthly return and benchmark monthly return           |
        |                                                                               |
        |           - Pie_chart_sector(): Return the generator of a Pie chart with the  |
        |               percentage of portfolio in each sector                          |
        |                                                                               |
        |           - Bar_plot_weights(): Return the generator of bar plot with the     |               
        |               weight of assets in the portfolio                               |
        |                                                                               |
        ---------------------------------------------------------------------------------
    """
    
    def __init__(self,start,end,actifs,ben,short_selling_allowed=True):
        
        from scipy.stats import skew, kurtosis
        self.n = len(actifs)
        self.ben = [ben]
        self.short_selling_allowed = short_selling_allowed
        self.actifs = actifs
        #_____________________________ Preparation of data _________________________________________________
        # Importation of assets
        np.random.seed(1)
        f = yf.download(actifs+self.ben, start=start, end=end)["Adj Close"].pct_change(1).dropna()
        self.f = pds.DataFrame(np.where(abs(f)>0.5,0,f),index=f.index, columns=f.columns)
        # Splittion train test
        split = int(0.7*len(f))
        self.f_train = self.f[actifs].iloc[:split,:]
        self.f_test = self.f[actifs].iloc[split:,:]
        self.ben_serie_train = self.f[self.ben].iloc[:split,:]
        self.ben_serie_test = self.f[self.ben].iloc[split:,:]
        
        
        #__________________________  Random weight simulation________________________________________________
        
        # Initialization list containor for the mean,std and weight of each simulation
        means = []
        std = []
        weights = []
        
        # Loop for simulation
        for _ in range(500):
            
            # Vector random number
            p = np.random.randint(-500,500,size=(len(self.f_train.columns),))
            
            # Normalize sum=1
            weight = (p/np.sum(p))
            
            # Compute portfolio returns
            pf = np.multiply(self.f_train,weight)
            sum_portfolio = pf.sum(axis=1)
            
            # Annualization and stocing
            means.append(sum_portfolio.mean()*252)
            std.append(sum_portfolio.std()*np.sqrt(252))
            weights.append(weight)
            
        # Ordering in datafrmame
        self.resume = pds.DataFrame([weights,means,std], index=["weights", "returns", "volatility"]).transpose()
        
    
    def max_sharpe_optimisation(self):
        
        from scipy.optimize import minimize

        # Initialisation weight value
        x0 = np.zeros(self.n)+(1/self.n)

        # Optimization constraints problem
        cons=({'type':'eq', 'fun': lambda x:sum(abs(x))-1})
        
        if self.short_selling_allowed == True:
            Bounds= [(-1 , 1) for i in range(0,self.n)]
        else:
            Bounds= [(0 , 1) for i in range(0,self.n)]

        # Lambda
        Lambda_RA=3

        # Optimization problem solving
        res_SR = minimize(self.SR_criterion, x0, method="SLSQP", args=(self.f_train, 0),bounds=Bounds,constraints=cons,options={'disp': False})

        # Result for visualization
        print(pds.DataFrame(np.round(res_SR.x,5),index=self.f_train.columns, columns=["Weihgt"]))

        # Result for computations
        self.X = res_SR.x
        
        self.sharpe = self.SR_criterion(self.X, self.f_train, 0)*np.sqrt(252)
        
    def mean_variance_optimisation(self):
        
        from scipy.optimize import minimize
        from scipy.stats import skew, kurtosis

        # Initialisation weight value
        x0 = np.zeros(self.n)+(1/self.n)

        # Optimization constraints problem
        cons=({'type':'eq', 'fun': lambda x:sum(abs(x))-1})
        if self.short_selling_allowed == True:
            Bounds= [(-1 , 1) for i in range(0,self.n)]
        else:
            Bounds= [(0 , 1) for i in range(0,self.n)]

        # Lambda
        Lambda_RA=3

        # Optimization problem solving
        res_EV = minimize(self.EV_criterion, x0, method="SLSQP", args=(3,self.f_train),bounds=Bounds,constraints=cons,options={'disp': False})

        # Result for visualization
        print(pds.DataFrame(np.round(res_EV.x,5),index=self.f_train.columns, columns=["Weihgt"]))

       
        # Result for computations
        self.X = res_EV.x
        
        self.sharpe = self.SR_criterion(self.X, self.f_train, 0)*np.sqrt(252)
        
    def mean_variance_skweness_kurtosis_optimisation(self):
        
        from scipy.optimize import minimize
        from scipy.stats import skew, kurtosis

        # Initialisation weight value
        x0 = np.zeros(self.n)+(1/self.n)

        # Optimization constraints problem
        cons=({'type':'eq', 'fun': lambda x:sum(abs(x))-1})
        if self.short_selling_allowed == True:
            Bounds= [(-1 , 1) for i in range(0,self.n)]
        else:
            Bounds= [(0 , 1) for i in range(0,self.n)]

        # Lambda
        Lambda_RA=3

        # Optimization problem solving
        res_EV = minimize(self.SK_criterion, x0, method="SLSQP", args=(3,self.f_train),bounds=Bounds,constraints=cons,options={'disp': False})

        # Result for visualization
        print(pds.DataFrame(np.round(res_EV.x,5),index=self.f_train.columns, columns=["Weihgt"]))

       
        # Result for computations
        self.X = res_EV.x
        
        self.sharpe = self.SR_criterion(self.X, self.f_train, 0)*np.sqrt(252)
        
    def max_sortino_optimisation(self):
        
        from scipy.optimize import minimize

        # Initialisation weight value
        x0 = np.zeros(self.n)+(1/self.n)

        # Optimization constraints problem
        cons=({'type':'eq', 'fun': lambda x:sum(abs(x))-1})
        if self.short_selling_allowed == True:
            Bounds= [(-1 , 1) for i in range(0,self.n)]
        else:
            Bounds= [(0 , 1) for i in range(0,self.n)]

        # Lambda
        Lambda_RA=3

        # Optimization problem solving
        res_SR = minimize(self.SOR_criterion, x0, method="SLSQP", args=(self.f_train, 0),bounds=Bounds,constraints=cons,options={'disp': False})

        # Result for visualization
        print(pds.DataFrame(np.round(res_SR.x,5),index=self.f_train.columns, columns=["Weihgt"]))

        # Result for computations
        self.X = res_SR.x
        
        
    def min_variance_optimisation(self):
        
        from scipy.optimize import minimize

        # Initialisation weight value
        x0 = np.zeros(self.n)+(1/self.n)

        # Optimization constraints problem
        cons=({'type':'eq', 'fun': lambda x:sum(abs(x))-1})
        if self.short_selling_allowed == True:
            Bounds= [(-1 , 1) for i in range(0,self.n)]
        else:
            Bounds= [(0 , 1) for i in range(0,self.n)]

        # Lambda
        Lambda_RA=3

        # Optimization problem solving
        res_SR = minimize(self.MV_criterion, x0, method="SLSQP", args=(self.f_train),bounds=Bounds,constraints=cons,options={'disp': False})

        # Result for visualization
        print(pds.DataFrame(np.round(res_SR.x,5),index=self.f_train.columns, columns=["Weihgt"]))

        # Result for computations
        self.X = res_SR.x
        
        
    def statistics(self):
       
        self.sharpe = -self.SR_criterion(self.X, self.f_test, 0)*np.sqrt(252)
        self.sortino = -self.SOR_criterion(self.X, self.f_test, 0)*np.sqrt(252)
        
        self.max_drawdown = -np.min(self.drawdown)
        self.pf = self.f_test.dot(self.X)
        
        self.beta = np.cov(self.ben_serie_test, self.pf, rowvar=False)[0][1]/np.var(self.ben_serie_test.values)
        self.alpha = np.mean(self.pf.values)*252 - self.beta*np.mean(self.pf.values)*252
        self.mean_portfolio = np.mean(self.f_train.values.dot(self.X))*252*100
        self.volatility_portfolio = np.std(self.f_train.values.dot(self.X))*np.sqrt(252)*100
        print(self.sharpe,self.mean_portfolio /self.volatility_portfolio)
        return self.sharpe, self.sortino



    
    def chart_cumulative_returns(self, train=True):
        import plotly.express as px

        if train==True:
            title = "Cumulative return on the train set"
            f = self.f_train
            b = self.ben_serie_train.cumsum()
        else:     
            title = "Cumulative return on the test set"
            f = self.f_test
            b = self.ben_serie_test.cumsum()
            
        pf = np.multiply(f,np.array(self.X))
        pf = pf.sum(axis=1).cumsum()
        self.pf = pds.DataFrame(pf,columns=["Portfolio cumulative returns"])
        
        self.bench = pds.DataFrame(b.values,columns=["Bench"])
        # Figure initialization
        fig = go.Figure()

        # Add first plot that represent the Cumulative return of the strategie
        fig.add_trace(go.Scatter(x=self.pf.index, y=self.pf["Portfolio cumulative returns"].values*100,
                          mode='lines',
                          name="Cumulative Return Strategie",
                          line=dict(color="#4E7EC1")))
        
        fig.add_trace(go.Scatter(x=self.pf.index, y=self.bench["Bench"].values*100,
                  mode='lines',
                  name="Bench",
                  line=dict(color="#C1574E")))
        
        fig.update_layout(title=title,
                   xaxis_title="Times",
                   yaxis_title="Cumulative Returns %", title_x=0.5,
                         legend=dict(
        x=0.03,
        y=0.97,
        traceorder='normal',
        font=dict(
            size=12,)))
        fig.show()
        return fig
        
    def previsions(self):
        
        values = []
        p = self.f_test.dot(self.X)
        for _ in range(500):
            values.append(np.cumsum(np.random.normal(p.mean(),p.std(),size=np.size(p))).reshape(-1,1))

        v = np.concatenate(values, axis=1)*100
        median = np.median(v,axis=1)
        centile_10 = np.percentile(v, q=10,axis=1)
        centile_90 = np.percentile(v, q=90,axis=1)

        y1_upper = centile_90
        y1_lower = centile_10
        y1_lower = y1_lower[::-1]


        x = [i for i in range(len(centile_10))]
        x_rev = x[::-1]
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x+x_rev,
            y=np.concatenate((y1_upper,y1_lower),axis=0),
            fill='toself',
            fillcolor='rgba(80,111,135,0.3)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name="confiance",
        ))


        fig.add_trace(go.Scatter(
            x=x, y=median,
            line_color="rgb(80,111,135)",
            name="median",
        ))
        fig.update_layout(title="Previsions return portfolio",
                   xaxis_title="Times",
                   yaxis_title="Cumulative Returns %", title_x=0.5,
                         legend=dict(
        x=0.03,
        y=0.97,
        traceorder='normal',
        font=dict(
            size=12,)))

        fig.update_traces(mode='lines')
        fig.show()
        return fig

    def chart_efficiency_borner(self):
        # Figure initialization
        fig = go.Figure()

        # Add first plot that represent the Cumulative return of the strategie
        fig.add_trace(go.Scatter(x=(self.resume["volatility"].values)*100, y=(self.resume["returns"].values)*100,
                          mode="markers",
                          name="Simulated Random Weight",
                                line=dict(color="#4E7EC1")))
        
        fig.add_trace(go.Scatter(x=[self.volatility_portfolio], y=[self.mean_portfolio],
                          mode="markers",
                          name="Optimal portfolio couple",
                                line=dict(color="#C1574E",width=15, dash='dot'),
                                marker={'size':15}))
        
        fig.update_layout(title="Efficiency frontier",
                   xaxis_title="Volatility in %",
                   yaxis_title="Returns in %",
                          title_x=0.5,
                         xaxis={"range":(0,100)},
                         yaxis={"range":(-15,100)},
                         legend=dict(
                        x=0.03,
                        y=0.97,
                        traceorder='normal',
                        font=dict(
                            size=12,)))
        fig.show()
        return fig
        
        
    def drawdown(self):
        p = self.f_test.dot(self.X)
        cum_rets = p.cumsum()+1

        # Calculate the running maximum
        running_max = np.maximum.accumulate(cum_rets.dropna())

        # Ensure the value never drops below 1
        running_max[running_max < 1] = 1

        # Calculate the percentage drawdown
        drawdown = (cum_rets)/running_max - 1
        self.drawdown = (cum_rets)/running_max - 1
        fig = go.Figure()


        y = np.concatenate((drawdown,np.array([0 for i in range(len(drawdown))])),axis=0)*100


        fig.add_trace(go.Scatter(
            x=drawdown.index.append(drawdown.index[::-1]), y=y,
            fill='toself',
            fillcolor="rgba(133,80,95,0.9)",
            line_color="rgba(133,80,95,0.9)",
            name="drawdown",
        ))

        fig.update_traces(mode='lines')
        
        fig.update_layout(title="Test set drawdown",
                   xaxis_title="Date",
                   yaxis_title="Drawdown %",
                          title_x=0.5,
                          legend=dict(
                        x=0.03,
                        y=0.97,
                        traceorder='normal',
                        font=dict(
                            size=12,))
                         )
        fig.show()
        return fig
        
    def heatmap_monthly_return(self):
        def profitable_month_return(p):
    
            total = 0
            positif = 0


            r=[]
            # Loop on each different year
            for year in p.index.strftime("%y").unique():
                e = []
                nbm = p.loc[p.index.strftime("%y")==year].index.strftime("%m").unique()
                # Loop on each different month
                for mois in nbm:

                    monthly_values =  p.loc[p.index.strftime("%y:%m")==f"{year}:{mois}"]
                    sum_ = monthly_values.sum()

                    # Verifying that there is at least 75% of the values
                    if len(monthly_values)>15:

                        # Computing sum return
                        s = monthly_values.sum()

                        if s>0:
                            positif+=1

                        else:
                            pass

                        total+=1

                    else:
                        pass
                    e.append(sum_)
                r.append(e)




            r[0]=[0 for _ in range(12-len(r[0]))] + r[0]
            r[-1]= r[-1]  + [0 for _ in range(12-len(r[-1]))] 
            return pds.DataFrame(r,columns=["January","February","March","April","May","June",
                                           "July","August","September","October","November","December"], index=p.index.strftime("%y").unique())

        r = profitable_month_return(self.f_test.dot(self.X))
        import plotly.figure_factory as ff

        colorscale=[[0.0, "#85505F"], [.5, "#507485"],[1.0, "#508561"]]


        fig = ff.create_annotated_heatmap(np.round(r*100,1).values,
                                         x = [i for i in r.columns],
                                         y = [f"20{i}" for i in r.index], 
                                         colorscale=colorscale,
                                         )
        fig.update_layout(title="Monthly return (test)",title_x=0.5
                         )
        
        fig.show()
        return fig
        
    def yearly_return_comparaison(self):
        def yearly_return_values(p):

            total = 0
            positif = 0


            r=[]
            # Loop on each different year
            for year in p.index.strftime("%y").unique():
                nbm = p.loc[p.index.strftime("%y")==year].index.strftime("%m").unique()
                # Loop on each different month
                for mois in nbm:

                    monthly_values =  p.loc[p.index.strftime("%y:%m")==f"{year}:{mois}"]
                    sum_ = monthly_values.sum()

                    # Verifying that there is at least 75% of the values
                    if len(monthly_values)>15:

                        # Computing sum return
                        s = monthly_values.sum()

                        if s>0:
                            positif+=1

                        else:
                            pass

                        total+=1

                    else:
                        pass
                    r.append(sum_)



            return r
        def yearly_return_index(p):

            total = 0
            positif = 0


            r=[]
            # Loop on each different year
            for year in p.index.strftime("%y").unique():
                e = []
                nbm = p.loc[p.index.strftime("%y")==year].index.strftime("%m").unique()
                # Loop on each different month
                for mois in nbm:

                    monthly_values =  p.loc[p.index.strftime("%y:%m")==f"{year}:{mois}"]
                    sum_ = monthly_values.sum()

                    # Verifying that there is at least 75% of the values
                    if len(monthly_values)>15:

                        # Computing sum return
                        s = monthly_values.sum()

                        if s>0:
                            
                            positif+=1

                        else:
                            pass

                        total+=1

                    else:
                        pass
                    e.append(sum_)
                r.append(e)




            r[0]=[0 for _ in range(12-len(r[0]))] + r[0]  
            
            r =  pds.DataFrame(r,columns=["January","February","March","April","May","June",
                                           "July","August","September","October","November","December"], index=p.index.strftime("%y").unique())

            v = []
            for i in [i for i in r.index]:
                for c in [i for i in r.columns]:
                    if r.loc[i,c]!=0:
                        v.append(f"{c}:20{i}")
            return v
        
        portfolio = self.f_test.dot(self.X)
        pf_values = yearly_return_values(portfolio)
        pf_index = yearly_return_index(portfolio)
        
        bench = pds.Series(self.ben_serie_test.values.flatten(), index=self.f_test.index)
        bench_values = yearly_return_values(bench)
        bench_index = yearly_return_index(bench)
        
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Portfolio", x=pf_index, y=pf_values, marker_line_color="#505B85",
                             marker_color="#508561", opacity=0.9))
        fig.add_trace(go.Bar(name="Bench", x=bench_index, y=bench_values, marker_line_color="#855061",
                             marker_color="#855061", opacity=0.9))
        # Change the bar mode
        fig.update_layout(barmode='group', title="Monthly return on the tets set", legend=dict(
                        x=0.03,
                        y=0.97,
                        traceorder='normal',
                        font=dict(
                            size=12,)),
                         title_x=0.5)
        fig.show()
        
        return fig
    
    def pie_chart_sector(self):
        
        s = sector.set_index("value")
        w = pds.DataFrame(self.X.transpose(), index=self.f_test.columns)
        e = pds.concat((w,s),axis=1).dropna()
        r = e.groupby("sector").sum()
        colors = ["#506F85", "#505485", "#665085", "#805085", "#85506F", "#618550", "#50855A", "#508574"]
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=[col for col in r.index], values=[v[0] for v in r.values], hole=0.35,
                            marker_colors=colors))
        fig.update_layout(
            title_text="Asset sector repartition ", title_x=0.5)
        return fig
    
    def bar_plot_weights(self):
       
        import plotly.express as px
        w = pds.DataFrame(self.X.transpose(), index=self.f_test.columns)
        fig = px.bar(x=[col for col in w.index], y=[value[0] for value in w.values],
                     text=[f"{np.round(value[0]*100,1)}%" for value in w.values])
        fig.update_traces(textposition='outside')
        fig.update_traces(marker_color="#4E7EC1", marker_line_color="#4E7EC1",
                  marker_line_width=1.5)
        fig.update_layout(
            title_text="Asset repartition portfolio ", title_x=0.5,legend=dict(
                        x=0.03,
                        y=0.97,
                        traceorder='normal',
                        font=dict(
                            size=12,)),
                    xaxis_title="Asset",
                   yaxis_title="Weight",)
        return fig
        
        
    
    def EV_criterion(self,weight,Lambda_RA,Returns_data):
        """ 
        ------------------------------------------------------------------------------
        | Output: optimization porfolio criterion                                    |
        ------------------------------------------------------------------------------
        | Inputs: -weight (type ndarray numpy): Wheight for portfolio                |
        |         -Lambda_RA (float):                                                |
        |         -Returns_data (type ndarray numpy): Returns of stocks              |
        ------------------------------------------------------------------------------
        """
        portfolio_return=np.multiply(Returns_data,np.transpose(weight));
        portfolio_return=np.sum(portfolio_return,1);
        mean_ret=np.mean(portfolio_return,0)
        sd_ret=np.std(portfolio_return,0)
        W=1;
        Wbar=1*(1+0.25/100);
        criterion=np.power(Wbar,1-Lambda_RA)/(1-Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)
        criterion=-criterion;
        return criterion

    
    def SR_criterion(self,weight, returns, rf):
        """ 
        ------------------------------------------------------------------------------
        | Output: Opposite Sharpe ratio to do a m imization                          |
        ------------------------------------------------------------------------------
        | Inputs: -Weight (type ndarray numpy): Wheight for portfolio                |
        |         -returns (type dataframe pandas): Returns of stocks                |
        |         -rf (float): risk-free asset                                       |
        ------------------------------------------------------------------------------
        """
        pf_return = returns.values.dot(weight)
        mu = np.mean(pf_return) - rf
        sigma = np.std(pf_return)
        Sharpe = -mu/sigma
        return Sharpe
    
    def SOR_criterion(self,weight, returns, rf):
        """ 
        ------------------------------------------------------------------------------
        | Output: Opposite Sharpe ratio to do a m imization                          |
        ------------------------------------------------------------------------------
        | Inputs: -Weight (type ndarray numpy): Wheight for portfolio                |
        |         -returns (type dataframe pandas): Returns of stocks                |
        |         -rf (float): risk-free asset                                       |
        ------------------------------------------------------------------------------
        """
        pf_return = returns.values.dot(weight)
        mu = np.mean(pf_return) - rf
        sigma = np.std(pf_return[pf_return<0])
        Sharpe = -mu/sigma
        return Sharpe
    
    def SK_criterion(self,weight,Lambda_RA,Returns_data):
        """ 
        ------------------------------------------------------------------------------
        | Output: optimization porfolio criterion                                    |
        ------------------------------------------------------------------------------
        | Inputs: -weight (type ndarray numpy): Wheight for portfolio                |
        |         -Lambda_RA (float):                                                |
        |         -Returns_data (type ndarray numpy): Returns of stocks              |
        ------------------------------------------------------------------------------
        """
        from scipy.stats import skew, kurtosis


        portfolio_return=np.multiply(Returns_data,np.transpose(weight));
        portfolio_return=np.sum(portfolio_return,1);
        mean_ret=np.mean(portfolio_return,0)
        sd_ret=np.std(portfolio_return,0)
        skew_ret=skew(portfolio_return,0)
        kurt_ret=kurtosis(portfolio_return,0)
        W=1;
        Wbar=1*(1+0.25/100);
        criterion=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)+Lambda_RA*(Lambda_RA+1)/(6)*np.power(Wbar,-2-Lambda_RA)*np.power(W,3)*skew_ret-Lambda_RA*(Lambda_RA+1)*(Lambda_RA+2)/(24)*np.power(Wbar,-3-Lambda_RA)*np.power(W,4)*kurt_ret
        criterion=-criterion;
        return criterion
    
    def MV_criterion(self,weight,Returns_data):
        """ 
        ------------------------------------------------------------------------------
        | Output: optimization porfolio criterion                                    |
        ------------------------------------------------------------------------------
        | Inputs: -weight (type ndarray numpy): Wheight for portfolio                |
        |         -Lambda_RA (float):                                                |
        |         -Returns_data (type ndarray numpy): Returns of stocks              |
        ------------------------------------------------------------------------------
        """
        portfolio_return=np.multiply(Returns_data,np.transpose(weight));
        portfolio_return=np.sum(portfolio_return,1);
        mean_ret=np.mean(portfolio_return,0)
        sd_ret=np.std(portfolio_return,0)
        criterion = sd_ret
        return criterion
    
    def forcing(self,weight):
        self.X = np.array(weight)
        

