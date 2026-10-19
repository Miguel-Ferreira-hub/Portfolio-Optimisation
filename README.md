
<h1 align="center">Portfolio Optimisation</h1>

<p align="center">
  <img src="Images/efficient_frontier.png" alt="Logo">
  <br>
  </b> Resulting portfolios visualised in expected returns - risk space. Optimisation is performed on three different objective functions; minimum variance, maximum returns and maximum Sharpe ratio following the procedure below on various U.S equities.
</p>

### 1. Expected Portfolio Return

The expected portfolio return is:

$$
\mathbb{E}[R_p] = \boldsymbol{w}^\top \boldsymbol{\mu}
= \sum_{i=1}^{n} w_i \mu_i
$$

where:

- \(w_i\) is the proportion of capital allocated to asset \(i\).
- \(\mu_i\) is the expected return of asset \(i\).
- \(n\) is the number of assets.

For historical daily returns, the expected returns can be estimated using the sample mean and annualised:

$$
\boldsymbol{\mu}_{\text{annual}}
= 252 \cdot \boldsymbol{\mu}_{\text{daily}}
$$

assuming 252 trading days per year.

### 2. Portfolio Variance and Volatility

Let \(\Sigma \in \mathbb{R}^{n \times n}\) be the covariance matrix of asset returns. Portfolio variance is:

$$
\sigma_p^2 = \boldsymbol{w}^\top
\boldsymbol{\Sigma}\boldsymbol{w}
$$

Portfolio volatility is the square root of variance:

$$
\sigma_p = \sqrt{\boldsymbol{w}^\top
\boldsymbol{\Sigma}\boldsymbol{w}}
$$

The covariance matrix captures both individual asset risk and the dependence between assets. Consequently, portfolio risk depends not only on the volatility of each asset but also on their correlations.

Using historical daily returns, annualised covariance is estimated as:

$$
\boldsymbol{\Sigma}_{\text{annual}}
= 252 \cdot \boldsymbol{\Sigma}_{\text{daily}}
$$

### 3. Portfolio Optimisation Objectives

Let \(r_f\) denote the annualised risk-free rate. Portfolio weights are subject to the constraints:

$$
\sum_{i=1}^{n} w_i = 1
$$

and, for a long-only portfolio with maximum allocation \(w_{\max}\),

$$
0 \leq w_i \leq w_{\max}
\qquad \forall i
$$

These constraints ensure that the portfolio is fully invested, with no short selling and a maximum allocation per asset.

#### 3.1 Maximum Sharpe Ratio

The Sharpe ratio measures excess return per unit of portfolio volatility:

$$
S(\boldsymbol{w}) =
\frac{\boldsymbol{w}^\top\boldsymbol{\mu} - r_f}
{\sqrt{\boldsymbol{w}^\top
\boldsymbol{\Sigma}\boldsymbol{w}}}
$$

The maximum Sharpe ratio portfolio is obtained by solving:

$$
\boxed{
\max_{\boldsymbol{w}}\;
\frac{\boldsymbol{w}^\top\boldsymbol{\mu} - r_f}
{\sqrt{\boldsymbol{w}^\top
\boldsymbol{\Sigma}\boldsymbol{w}}}
}
$$

This objective identifies the portfolio offering the highest expected excess return per unit of risk.

#### 3.2 Minimum Variance

The minimum variance portfolio minimises total portfolio risk:

$$
\boxed{
\min_{\boldsymbol{w}}\;
\boldsymbol{w}^\top
\boldsymbol{\Sigma}\boldsymbol{w}
}
$$

subject to the portfolio constraints.

This portfolio focuses exclusively on risk minimisation and does not directly optimise expected return. It is useful when the primary objective is to construct the least volatile portfolio within the permitted investment universe.

#### 3.3 Maximum Expected Return

The maximum expected return portfolio solves:

$$
\boxed{
\max_{\boldsymbol{w}}\;
\boldsymbol{w}^\top\boldsymbol{\mu}
}
$$

subject to the portfolio constraints.

This objective allocates capital to maximise expected return without directly penalising portfolio volatility. Under long-only allocation bounds, the solution tends to favour assets with the highest estimated expected returns, subject to the permitted allocation limits.

### 4. Summary of Optimisation Objectives

| Objective | Optimisation function | Primary goal |
|---|---|---|
| Maximum Sharpe | Maximise \((R_p-r_f)/\sigma_p\) | Maximise risk-adjusted return |
| Minimum variance | Minimise \(\boldsymbol{w}^\top\Sigma\boldsymbol{w}\) | Minimise portfolio risk |
| Maximum return | Maximise \(\boldsymbol{w}^\top\boldsymbol{\mu}\) | Maximise expected return |

### 5. Numerical Optimisation

The portfolio optimisation problem can be solved numerically using constrained optimisation algorithms such as Sequential Least Squares Programming (SLSQP).

The optimisation procedure is:

1. Estimate expected asset returns and the covariance matrix from historical price data.
2. Annualise the estimates consistently.
3. Define the portfolio constraints, including full investment and individual asset bounds.
4. Select the desired objective function.
5. Solve for the optimal portfolio weights.
6. Calculate the resulting expected return, volatility and Sharpe ratio.

The resulting portfolios can be compared in risk-return space and used to construct the efficient frontier, illustrating the trade-off between expected return and portfolio risk.

<h1 align="center">Theoretical Paths</h1>

The following plot displays the theoretical wealth paths for the optimised portfolio under the different objective functions in retrospect (likely to fail out of sample due to overfitting).

<p align="center">
  <img src="Images/portfolio_paths.png" alt="Logo">
</p>
