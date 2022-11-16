import pandas as pd 
import polars as pl
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from lib.baseClass import BaseClass
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import black_litterman, risk_models, expected_returns, objective_functions
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import plotting

class PortfolioManager(BaseClass):

    def __init__(self, investmet=40000, verbose=False, save=True):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.save = save
        self.investment = investmet

    def get_portfolio(self, df, gamma=0):
        
        # Calculate expected returns and sample covariance
        returns = expected_returns.returns_from_prices(df)
        returns = returns.clip(returns.quantile(0.01), returns.quantile(0.99), axis=1)
        mu = returns.mean(axis=0)*252

        S = risk_models.risk_matrix(df, method="ledoit_wolf_constant_correlation")
        cleaned_weights = self.efficient_frontier(mu, S, gamma, "efficient_risk", p=0.20)

        cont, disc = self.process_weights(df, cleaned_weights)
        self.plot_results(df,cont,disc)

        return cont, disc
    
    def efficient_frontier(self, mu, S, gamma, mode, p=None):
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S, solver="CVXOPT",
                              verbose=self.verbose, weight_bounds=(0,1))
        if gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=gamma)
        if mode == "max_sharpe":
            ef.max_sharpe()
        if mode == "efficient_return":
            ef.efficient_return(p)   
        if mode == "efficient_risk":
            ef.efficient_risk(p)
        if mode == "max_quadratic_utility":
            ef.max_quadratic_utility()  
        if self.verbose:
            ef.portfolio_performance(verbose=True)

        return ef.clean_weights()

    def risk_parity(self, df, verbose=False):
        historical_returns = expected_returns.returns_from_prices(df)
        S = risk_models.risk_matrix(df, method="ledoit_wolf_constant_correlation")

        # Optimize for maximal Sharpe ratio
        hrp = HRPOpt(historical_returns, S)

        raw_weights = hrp.optimize()
        cleaned_weights = hrp.clean_weights()
        cont = pd.DataFrame.from_dict(cleaned_weights, orient="index")
        cont.reset_index(inplace=True)
        cont.columns = ["symbol", "weights"]
        if verbose:
            plotting.plot_dendrogram(hrp)
            plt.show()
            
        return cleaned_weights, cont

    def bl(self, df, gamma=0, absolute_views=None):

        start, end = df.index[0], df.index[-1]
        bench = pd.read_csv("D:/Data/SP500_D1.csv",index_col=0)
        bench.index = pd.to_datetime(bench.index)
        market_prices = bench.loc[start:end, 'close']

        n = len(df.columns)
        mcaps = {col: 1/n for col in df.columns}
        S = risk_models.risk_matrix(df, method="ledoit_wolf_constant_correlation")
        delta = black_litterman.market_implied_risk_aversion(market_prices)
        prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

        mu = expected_returns.mean_historical_return(df, frequency=252)
        views = {col: mu.loc[col] for col in df.columns}
        bl = BlackLittermanModel(S, pi=prior, absolute_views=views)

        rets = bl.bl_returns()
        cleaned_weights = self.efficient_frontier(rets, S, gamma, "max_sharpe")

        cont, disc = self.process_weights(df, cleaned_weights)
        self.plot_results(df,cont,disc)

        return cont, disc
    

    def process_weights(self, df, cleaned_weights):

        cont = pd.DataFrame.from_dict(cleaned_weights, orient="index")
        cont.reset_index(inplace=True)
        cont.columns = ["symbol", "weights"]

        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(cleaned_weights, latest_prices, self.investment)
        allocation, leftover = da.greedy_portfolio()
        disc = pd.DataFrame(allocation, index=[0]).T
        disc.reset_index(inplace=True)
        disc.columns = ["symbol", "weights"]
        disc["latest_price"] = np.round(latest_prices.loc[disc.symbol].values, 2)
        disc["investment"] = latest_prices.loc[disc.symbol].values * disc.weights.values
        disc["percentage"] = disc["investment"].values / self.investment

        infos = pl.read_csv(f"D:/EOD/Temp/available_files.csv").to_pandas()
        infos.set_index("symbol", inplace=True)
        infos = infos.loc[disc.symbol] 

        disc = pd.concat([infos, disc.set_index("symbol")], axis=1)
        disc.reset_index(inplace=True)
        disc = disc.loc[:, ["symbol", "weights", "latest_price",
                            "investment", "percentage", "type", "name", "isin"]]
        disc.sort_values("investment", ascending=False, inplace=True)

        print(f"Number of Items continuous: {(cont.weights > 0).sum()}")
        print(f"Number of Items discrete: {(disc.weights > 0).sum()}")

        if self.save:
            self.create_folder(f"Results")
            cont.to_csv(f"{self.path}/Results/continuous_weights.csv")
            disc.to_csv(f"{self.path}/Results/discrete_weights.csv", index=False)
        
        return cont, disc
    
    def plot_results(self, df, cont, disc, report=True):

        e = 1e-8
        startdate, enddate = df.index[0], df.index[-1]
        bench = pd.read_csv("D:/SP500_D1.csv")
        bench.set_index(pd.to_datetime(bench.date), inplace=True)
        bench = bench.loc[startdate:enddate, "close"] 
        bench = pd.DataFrame(bench / bench.iloc[0] * 100)
        bench.columns = ["SP500"]
        
        disc = df.loc[:, disc.symbol.to_numpy()].values @ disc.weights.values
        returns = pd.Series(disc, name="Close").pct_change()
        returns.index = df.index
        disc = pd.DataFrame((disc/disc[0]+e) * 100,
                            columns=["Optimized Portfolio"])
        disc['date'] = df.index
        disc.set_index('date', inplace=True)

        data = df.iloc[:, 1:]
        eq = data.values @ np.ones((np.shape(data)[1], 1))
        eq = pd.DataFrame((eq/eq[0]+e) * 100, columns=["1/n"])
        eq['date'] = df.index
        eq.set_index('date', inplace=True)

        ax = bench.plot()
        disc.plot(ax=ax)
        eq.plot(ax=ax)
        fig = ax.get_figure()

        if self.save:
            self.create_folder(f"Results")
            fig.savefig(f"{self.path}/Results/Performances.png")

        if report:
            qs.reports.html(returns, bench,
                            output=f"{self.path}/Results/Report.html")

if __name__ == "__main__":

    pm = PortfolioManager(verbose=True)
    df = pl.read_parquet("D:/EOD/Datasets/ETF_1440_pivot.parquet").to_pandas()
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    print(df.info())

    pm.get_portfolio(df)
 
