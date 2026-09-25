# =============================================================================
#                               IMPORTS
# =============================================================================

import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import scipy.optimize as sp

# =============================================================================
#                               FUNCTIONS
# =============================================================================

# Calculate returns
def compute_returns(df):
    returns = df.pct_change(fill_method=None).dropna()
    return returns

# Compute Expected Returns and Covariance
def compute_statistics(df):
    expected_returns = df.mean() * 252
    covariance = df.cov() * 252
    condition_number = np.linalg.cond(covariance)
    print(f"Covariance Matrix Condition Number: {condition_number:.2f}")
    if condition_number > 1e5:
        print("WARNING COVARIANCE MATRIX IS ILL-CONDITIONED (UNSTABLE)")
    return expected_returns, covariance

# Apply Weights to a Portfolio
def generate_portfolio(weights,expected_returns,covariance,rf=0.0452):
    portfolio_returns = weights @ expected_returns 
    portfolio_variance = weights @ covariance @ weights
    risk = np.sqrt(portfolio_variance)
    sharpe = (portfolio_returns - rf) / risk
    return portfolio_returns,risk,sharpe

# Generate Random Portfolios (Monte Carlo)
# def generate_random_portfolio(expected_returns,covariance,rf=0.0452):
#     weights = np.random.random(expected_returns.shape[0])
#     weights /= weights.sum() # Normalise weights (add up to 1)
#     portfolio_returns = weights @ expected_returns 
#     portfolio_variance = weights @ covariance @ weights
#     risk = np.sqrt(portfolio_variance)
#     sharpe = (portfolio_returns - rf) / risk # Optimisation objective function for tangency portfolio
#     return portfolio_returns,risk,sharpe

# Generate Individual Asset Portfolio
def generate_individual_asset_portfolio(expected_returns,covariance,rf=0.0452):
    weights = np.identity(expected_returns.shape[0]) # 1 for each asset 0 elsewhere
    portfolio_returns = weights @ expected_returns 
    portfolio_variance = weights @ covariance @ weights
    risk = np.sqrt(np.diag(portfolio_variance))
    sharpe = (portfolio_returns - rf) / risk
    return portfolio_returns,risk,sharpe

# =============================================================================
#                               OPTIMISATION FUNCTION HELPERS
# =============================================================================

# Compute Weight Normalisation Constraint (add to 1)
def compute_constraint(weights):
    con = weights.sum() - 1
    return con

# Compute Optimisation Objective - Maximise Sharpe Ratio
def compute_sharpe_objective(weights,expected_returns,covariance,rf=0.0452):
    portfolio_returns = weights @ expected_returns 
    portfolio_variance = weights @ covariance @ weights
    risk = np.sqrt(portfolio_variance)
    sharpe = (portfolio_returns - rf) / risk # Optimisation objective function for tangency portfolio
    return -sharpe

# Compute Optimisation Objective - Minimise Variance
def compute_variance_objective(weights,covariance):
    variance = weights @ covariance @ weights
    return variance

# Compute Optimisation Objective - Maximise Returns
def compute_returns_objective(weights,expected_returns):
    portfolio_returns = weights @ expected_returns 
    return -portfolio_returns

# Compute Portfolio Metrics
def compute_portfolio_info(tickers,weights,expected_returns,covariance,info: str):
    # Maximum sharpe ratio portfolio
    portfolio_return,portfolio_risk,sharpe = generate_portfolio(weights=weights,expected_returns=expected_returns,covariance=covariance)

    # Print statements
    print(f"------Optimal Weights ({info})------")
    for ticker,weight in zip(tickers,weights):
        print(f"{ticker}: {weight:.2f}")
    print(f"Portfolio Return (%): {portfolio_return*100:.2f}")
    print(f"Porfolio Risk (σ): {portfolio_risk:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    return portfolio_return,portfolio_risk,sharpe

# Compute Portfolio Returns Constraint
def compute_portfolio_return(weights,expected_returns):
    returns = weights @ expected_returns
    return returns

# Optimise Portfolio
def compute_optimisation(compute_objective,compute_constraint,args,target=None,type="normal"):
    x0 = np.ones(len(tickers)) / len(tickers)

    # Max allocation of 15% per asset
    b = []
    for _ in range(len(tickers)):
        b.append(((0,0.15)))

    bounds = tuple(b)

    if type == "normal":
        con = {'type':'eq','fun':compute_constraint}
        cons = [con]
    elif (type == "frontier") and (target != None):
        weight_con = {'type':'eq','fun':compute_constraint}
        target_con = {"type":"eq","fun": lambda w, target=target:
                compute_portfolio_return(weights=w,expected_returns=expected_returns) - target
        }
        cons = [weight_con] + [target_con]

    sol = sp.minimize(compute_objective,x0,args=args,method='SLSQP',bounds=bounds,constraints=cons)

    if not sol.success:
        print(f"Optimisation failed: {sol.message}")
        return None

    return sol.x

# Construct Optimised Portfolio With New Weights
def construct_portfolio(weights,data,tickers,investment):
    portfolio = None
    for ticker, weight in zip(tickers,weights):
        returns = data[ticker].pct_change().dropna()
        growth = (1 + returns).cumprod()
        value = investment*weight*growth
        if portfolio is None:
            portfolio = value
        else:
            portfolio = portfolio.add(value, fill_value=0)
    start_date = data.index.min() - pd.DateOffset(days=1)
    portfolio = pd.concat([pd.Series([1],index=[start_date]),portfolio])
    return portfolio

# Compute Market Capital Line
def compute_tangent(tangency_returns,tangency_risk,efficient_frontier_risk,rf=4.52):
    tangency_returns *= 100
    m = (tangency_returns - rf)/tangency_risk
    c = rf
    domain = np.linspace(0,max(efficient_frontier_risk),len(efficient_frontier_risk))
    tangent = m*domain + c
    return tangent,domain

# =============================================================================
#                               DATA AND TICKER LISTS
# =============================================================================

end = dt.datetime.now()
start = end - dt.timedelta(days=2*365)
interval = '1d'

# First 100 tickers of S&P500
tickers = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX",
    "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BE", "BNY", "BA",
    "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BG", "BXP", "CHRW",
    "CDNS", "CPT", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY", "CAT", "CBOE",
    "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX"
]

# Collection of random equities and bonds
# tickers = [
#     # =========================
#     # Equities
#     # =========================
#     "AAPL",   # Apple
#     "MSFT",   # Microsoft
#     "AMZN",   # Amazon
#     "GOOGL",  # Alphabet
#     "NVDA",   # NVIDIA
#     "META",   # Meta
#     "AVGO",   # Broadcom
#     "AMD",    # AMD
#     "JPM",    # JPMorgan
#     "BRK-B",  # Berkshire Hathaway
#     "JNJ",    # Johnson & Johnson
#     "LLY",    # Eli Lilly
#     "PG",     # Procter & Gamble
#     "KO",     # Coca-Cola
#     "WMT",    # Walmart
#     "HD",     # Home Depot
#     "CAT",    # Caterpillar
#     "XOM",    # Exxon Mobil
#     "CVX",    # Chevron
#     "UNH",    # UnitedHealth

#     # =========================
#     # Bonds / Bond ETFs
#     # =========================
#     "BND",    # Vanguard Total Bond Market
#     "AGG",    # iShares Core US Aggregate Bond
#     "BNDX",   # Vanguard Total International Bond
#     "SHY",    # 1-3 Year US Treasury
#     "IEI",    # 3-7 Year US Treasury
#     "IEF",    # 7-10 Year US Treasury
#     "TLT",    # 20+ Year US Treasury
#     "SGOV",   # 0-3 Month US Treasury
#     "BIL",    # 1-3 Month T-Bills
#     "TIP",    # US TIPS
#     "LQD",    # Investment Grade Corporate Bonds
#     "HYG",    # High Yield Corporate Bonds
#     "VCIT",   # Intermediate Corporate Bonds
#     "VCSH",   # Short-Term Corporate Bonds
#     "VCLT",   # Long-Term Corporate Bonds
#     "MUB",    # Municipal Bonds
#     "VTEB",   # Tax-Exempt Municipal Bonds
#     "EMB",    # Emerging Market Bonds
#     "GOVT",   # US Treasury Bonds
#     "VGIT"    # Intermediate-Term Treasury
# ]

# Portfolio consisting of many diversified assets
# tickers = [
#     # ============================================================
#     # 100 EQUITIES
#     # ============================================================
#     "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
#     "GOOG", "META", "AVGO", "BRK-B", "JPM",
#     "LLY", "WMT", "XOM", "V", "JNJ",
#     "MA", "COST", "ORCL", "NFLX", "HD",
#     "PG", "KO", "CSCO", "IBM", "GE",
#     "CAT", "RTX", "BA", "UNH", "MRK",
#     "ABBV", "PFE", "TMO", "ABT", "DHR",
#     "ACN", "CRM", "ADBE", "AMD", "QCOM",
#     "TXN", "AMAT", "MU", "INTC", "NOW",
#     "INTU", "PLTR", "PANW", "CRWD", "UBER",
#     "TSLA", "NKE", "MCD", "SBUX", "LOW",
#     "TGT", "TJX", "BKNG", "ABNB", "DIS",
#     "CMCSA", "VZ", "COP", "CVX", "SLB",
#     "EOG", "OXY", "GS", "MS", "BAC",
#     "WFC", "C", "SCHW", "BLK", "BX",
#     "SPGI", "CME", "ICE", "CB", "AON",
#     "PGR", "ALL", "TRV", "LIN","APD", 
#     "NEE", "DUK", "DE", "HON","LMT", 
#     "GD", "ADP", "PYPL", "SHOP","MRVL", 
#     "KLAC", "LRCX", "CSX", "UNP",

#     # ============================================================
#     # 40 ETFs
#     # ============================================================
#     # Broad equity
#     "SPY", "QQQ", "IWM", "DIA", "VTI",
#     "VOO", "VUG", "VTV", "RSP",

#     # International equity
#     "EFA", "EEM", "VXUS", "VEA", "VWO",

#     # Commodities
#     "GLD", "IAU", "SLV", "USO", "UNG",
#     "DBA", "PDBC",

#     # Bonds
#     "BND", "AGG", "TLT", "IEF", "SHY",
#     "SGOV", "BIL", "TIP", "LQD", "HYG",
#     "VCIT", "VCSH", "VCLT", "EMB", "BNDX",
#     "MUB",

#     # Real estate / sectors
#     "VNQ", "XLK", "XLF",

#     # ============================================================
#     # 30 FUTURES
#     # ============================================================
#     # Equity index futures - Warning code not designed to handle futures
#     # "ES=F",       # E-mini S&P 500
#     # "MES=F",      # Micro E-mini S&P 500
#     # "NQ=F",       # E-mini Nasdaq-100
#     # "MNQ=F",      # Micro E-mini Nasdaq-100
#     # "YM=F",       # E-mini Dow
#     # "MYM=F",      # Micro E-mini Dow
#     # "RTY=F",      # E-mini Russell 2000
#     # "M2K=F",      # Micro E-mini Russell 2000

#     # # Treasury futures
#     # "ZB=F",       # US Treasury Bond
#     # "ZN=F",       # 10-Year Treasury Note
#     # "ZF=F",       # 5-Year Treasury Note
#     # "ZT=F",       # 2-Year Treasury Note

#     # # Precious metals
#     # "GC=F",       # Gold
#     # "MGC=F",      # Micro Gold
#     # "SI=F",       # Silver
#     # "SIL=F",      # Micro Silver
#     # "PL=F",       # Platinum
#     # "PA=F",       # Palladium

#     # # Energy
#     # "CL=F",       # WTI Crude Oil
#     # "MCL=F",      # Micro WTI
#     # "BZ=F",       # Brent Crude
#     # "NG=F",       # Natural Gas
#     # "QG=F",       # E-mini Natural Gas

#     # # Industrial / agricultural
#     # "HG=F",       # Copper
#     # "ZC=F",       # Corn
#     # "ZS=F",       # Soybeans
#     # "ZW=F",       # Wheat
#     # "ZM=F",       # Soybean Meal

#     # # FX futures
#     # "6E=F",       # Euro
#     # "6J=F",       # Japanese Yen

#     # ============================================================
#     # 15 FX
#     # ============================================================
#     "EURUSD=X",
#     "GBPUSD=X",
#     "JPY=X",
#     "CHF=X",
#     "CAD=X",
#     "AUDUSD=X",
#     "NZDUSD=X",
#     "CNY=X",
#     "HKD=X",
#     "SEK=X",
#     "NOK=X",
#     "SGD=X",
#     "INR=X",
#     "MXN=X",
#     "BRL=X",

#     # ============================================================
#     # 15 CRYPTO
#     # ============================================================
#     "BTC-USD",
#     "ETH-USD",
#     "BNB-USD",
#     "XRP-USD",
#     "SOL-USD",
#     "ADA-USD",
#     "DOGE-USD",
#     "AVAX-USD",
#     "DOT-USD",
#     "LINK-USD",
#     "LTC-USD",
#     "BCH-USD",
#     "XLM-USD",
#     "TRX-USD"
# ]

data = yf.download(tickers=tickers,start=start,end=end,interval=interval)
returns = compute_returns(data['Close'])
expected_returns, covariance = compute_statistics(returns)

# =============================================================================
#                               OPTIMISATION CALLS
# =============================================================================

# Mean-Variance Optimisation (MVO)
weights_max_sharpe = compute_optimisation(compute_objective=compute_sharpe_objective,
                                            compute_constraint=compute_constraint,
                                            args=(expected_returns,covariance))

# Minimum Variance Portfolio (MVP)
weights_min_variance = compute_optimisation(compute_objective=compute_variance_objective,
                                                compute_constraint=compute_constraint,
                                                args=(covariance,))

# Maximum Return Portfolio
weights_max_returns = compute_optimisation(compute_objective=compute_returns_objective,
                                            compute_constraint=compute_constraint,
                                            args=(expected_returns,))

# Maximum Sharpe Ratio Portfolio
portfolio_return_max_sharpe,portfolio_risk_max_sharpe,max_sharpe = compute_portfolio_info(tickers=tickers,
                                                                                            weights=weights_max_sharpe,
                                                                                            expected_returns=expected_returns,
                                                                                            covariance=covariance,info="Max Sharpe")

portfolio_return_min_variance,portfolio_risk_min_variance,sharpe_min_variance = compute_portfolio_info(tickers=tickers,
                                                                                            weights=weights_min_variance,
                                                                                            expected_returns=expected_returns,
                                                                                            covariance=covariance,info="Min Variance")

portfolio_return_max_returns,portfolio_risk_max_returns,sharpe_max_returns = compute_portfolio_info(tickers=tickers,
                                                                                            weights=weights_max_returns,
                                                                                            expected_returns=expected_returns,
                                                                                            covariance=covariance,info="Max Returns")


# Individual Assets
individual_asset_returns,individual_asset_risk,individual_asset_sharpe = generate_individual_asset_portfolio(expected_returns=expected_returns,covariance=covariance)

# Produce Efficient Frontier
target_returns = np.linspace(np.min(expected_returns),portfolio_return_max_returns,100)
frontier_returns = []
frontier_risk = []
frontier_weights = []
frontier_sharpe = []

for target in target_returns:
    weights = compute_optimisation(compute_objective=compute_variance_objective,compute_constraint=compute_constraint,args=(covariance,),target=target,type="frontier")

    if weights is None:
        continue

    returns,risk,sharpe = generate_portfolio(weights=weights,expected_returns=expected_returns,covariance=covariance)
    frontier_returns.append(returns*100)
    frontier_risk.append(risk)
    frontier_sharpe.append(sharpe)
    frontier_weights.append(weights)

# Produce Tangent Line (Capital Market Line) and Sharpe Colourbar Limits
tangent,domain = compute_tangent(portfolio_return_max_sharpe,portfolio_risk_max_sharpe,frontier_risk)
sharpe_min = np.min(individual_asset_sharpe)
sharpe_max = np.max(individual_asset_sharpe)

# Monte Carlo Simulation for Randomly Initialised Weights
# n_sims = 100
# risk_list = []
# portfolio_returns_list = []
# sharpe_list = []

# for i in tqdm(range(n_sims),desc='Generating Monte Carlo Portfolios'): # Keep track of when it is done
#     portfolio_returns,risk,sharpe = generate_random_portfolio(expected_returns=expected_returns,covariance=covariance)
#     portfolio_returns_list.append(portfolio_returns*100)
#     risk_list.append(risk)
#     sharpe_list.append(sharpe)

# Portfolio History as if was Optimised on Start (Overfit Backtest)
investment = 1

portfolio_max_sharpe = construct_portfolio(weights=weights_max_sharpe,data=data['Close'],tickers=tickers,investment=investment)
portfolio_min_variance = construct_portfolio(weights=weights_min_variance,data=data['Close'],tickers=tickers,investment=investment)
portfolio_max_returns = construct_portfolio(weights=weights_max_returns,data=data['Close'],tickers=tickers,investment=investment)

# =============================================================================
#                               VISUALISATIONS AND PLOTS
# =============================================================================

fig = go.Figure()

# fig.add_trace(
#     go.Scatter(
#         x=risk_list,
#         y=portfolio_returns_list,
#         mode="markers",name="",
#         marker=dict(
#         size=10,
#         color=sharpe_list,
#         colorscale="Viridis",
#         colorbar=dict(title="Sharpe Ratio"))
#         )
#     )

fig.add_trace(
    go.Scatter(
        x=individual_asset_risk,
        y=individual_asset_returns*100,
        mode='markers+text',
        text=tickers,
        textposition='top center',
        name="",
        showlegend=False,
        textfont=dict(
            size=7
        ),
        marker=dict(
            size=10,
            color=individual_asset_sharpe,
            colorscale="Viridis",
            cmin=sharpe_min,
            cmax=sharpe_max,
            colorbar=dict(
                title="Sharpe Ratio"
            )
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[portfolio_risk_max_sharpe],
        y=[portfolio_return_max_sharpe*100],
        mode="markers",
        name=f"Tangency Portfolio (Max Sharpe -> {max_sharpe:.2f})",
        marker=dict(
        size=10)
        )
    )

fig.add_trace(
    go.Scatter(
        x=[portfolio_risk_min_variance],
        y=[portfolio_return_min_variance*100],
        mode="markers",
        name=f"Minimum Variance Portfolio",
        marker=dict(
        size=10)
        )
    )

fig.add_trace(
    go.Scatter(
        x=[portfolio_risk_max_returns],
        y=[portfolio_return_max_returns*100],
        mode="markers",
        name=f"Maximum Returns Portfolio",
        marker=dict(
        size=10)
        )
    )

fig.add_trace(
    go.Scatter(
        x=frontier_risk,
        y=frontier_returns,
        mode="lines",name=f"Efficient Frontier",
        )
    )

fig.add_trace(
    go.Scatter(
        x=domain,
        y=tangent,
        mode="lines",name=f"Capital Market Line",
        )
    )

fig.update_yaxes(title="Expected Returns (%)",ticksuffix="%")
fig.update_xaxes(title="Portfolio Risk (σ)")
fig.update_layout(template="plotly_dark",height=900,
    title=dict(
        text="Portfolio Optimisation",x=0.5,xanchor="center"),
    yaxis_tickformat='.2r')

fig.update_layout(
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top"
    )
)

fig.show()

# Portfolio Paths
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=portfolio_max_sharpe,
        mode="lines",
        name=f"Maximum Sharpe Portfolio",
        marker=dict(
        size=10)
        )
    )

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=portfolio_min_variance,
        mode="lines",
        name=f"Minimum Variance Portfolio",
        marker=dict(
        size=10)
        )
    )

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=portfolio_max_returns,
        mode="lines",
        name=f"Maximum Returns Portfolio",
        marker=dict(
        size=10)
        )
    )

fig.update_yaxes(title="Portfolio Paths ($)",tickprefix="$")
fig.update_xaxes(title="Date")
fig.update_layout(template="plotly_dark",height=900,
    title=dict(
        text="Portfolio Paths",x=0.5,xanchor="center"),
    yaxis_tickformat='.2r')

fig.show()
