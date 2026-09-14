from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd

MIN_OBSERVATIONS = 150
MIN_OPEN_INTEREST = 3000
def select_contracts(panel, start_year=2013, delivery_months=(3, 7, 12)):
    """
    Restrict to the sample the test should run on.

    start_year is 2013 because before that the CBOT print at the SAFEX close
    is stale on 63% of days, which triples the negative lag-1 autocorrelation
    of changes. That is manufactured mean reversion and would contaminate
    exactly the parameter being estimated.

    The open interest floor drops contracts where the JSE mark is often
    exchange-derived rather than traded, so it is not a real price.
    """
    frame = panel.copy()
    frame["dt"] = pd.to_datetime(frame["trade_date"])
    frame["delivery_month"] = pd.to_datetime(frame["expiry"]).dt.month

    frame = frame[
        (frame["dt"].dt.year >= start_year)
        & (frame["delivery_month"].isin(delivery_months))
        & (frame["safex_open_interest"] >= MIN_OPEN_INTEREST)
    ]
    return frame.sort_values(["expiry", "dt"])

# Function for ADF test per contract expiry

def adf_by_contract(frame, column):
    """ADF within each contract. Returns one row per contract."""
    rows = []
    for expiry, group in frame.groupby("expiry"):
        series = group[column].dropna()
        if len(series) < MIN_OBSERVATIONS:
            continue

        # regression="c" allows a non-zero mean but no time trend. The arb
        # should have a level (the cost stack) but no reason to drift.
        result = adfuller(series, regression="c", autolag="AIC", result_object=True)

        rows.append({
            "expiry": expiry,
            "delivery_month": group["delivery_month"].iloc[0],
            "n": len(series),
            "adf_stat": result.statistic,
            "p_value": result.pvalue,
        })
    return pd.DataFrame(rows)

# Function for Fisher test
def fisher_combine(pvalues):
    """
    Maddala-Wu: -2 * sum(log p) is chi-squared with 2N degrees of freedom
    under the joint null that EVERY contract has a unit root.

    Rejecting means at least some contracts are stationary, not all of them.
    """
    clean = pvalues.dropna().clip(lower=1e-10)
    statistic = -2.0 * np.log(clean).sum()
    degrees_of_freedom = 2 * len(clean)
    return {
        "n_contracts": len(clean),
        "fisher_stat": statistic,
        "df": degrees_of_freedom,
        "p_value": stats.chi2.sf(statistic, degrees_of_freedom),
    }

# Function for the first stage of Engle-Granger per contract expiry

def engle_granger_by_contract(frame, dependent="log_safex", independent="log_cbot"):
    """
    Regress one leg on the other within each contract, keeping the residual.

    This is the first stage of Engle-Granger. It lets the data choose the
    ratio instead of imposing 1, which is what the arb does. In logs, beta is
    an elasticity: beta = 0.5 means a 10% move in corn is associated with a
    5% move in SAFEX.
    """
    rows = []
    residuals = {}

    for expiry, group in frame.groupby("expiry"):
        clean = group[[dependent, independent]].dropna()
        if len(clean) < MIN_OBSERVATIONS:
            continue

        design = sm.add_constant(clean[independent])
        fit = sm.OLS(clean[dependent], design).fit()

        residuals[expiry] = fit.resid
        rows.append({
            "expiry": expiry,
            "delivery_month": group["delivery_month"].iloc[0],
            "n": len(clean),
            "alpha": fit.params.iloc[0],
            "beta": fit.params.iloc[1],
            "r_squared": fit.rsquared,
        })

    return pd.DataFrame(rows), residuals

# Function for the cointegration test per contract expiry

def coint_by_contract(frame, dependent="log_safex", independent="log_cbot"):
    """
    Engle-Granger cointegration test with the CORRECT critical values.

    coint() runs the regression and tests the residual using MacKinnon's
    distribution for a fitted residual, not the standard Dickey-Fuller one.
    Using adfuller() on the residual would over-reject.
    """
    rows = []
    for expiry, group in frame.groupby("expiry"):
        clean = group[[dependent, independent]].dropna()
        if len(clean) < MIN_OBSERVATIONS:
            continue

        statistic, pvalue, critical = coint(
            clean[dependent], clean[independent], trend="c", autolag="AIC"
        )
        rows.append({
            "expiry": expiry,
            "delivery_month": group["delivery_month"].iloc[0],
            "n": len(clean),
            "coint_stat": statistic,
            "p_value": pvalue,
        })
    return pd.DataFrame(rows)
