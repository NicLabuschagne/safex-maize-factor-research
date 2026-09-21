"""regimes.py — empirical regime bucketing for a single price series.

This is a *descriptive* study, not a forecasting model. The question it answers
is narrow: does the YM return process separate into a small number of stable,
persistent buckets that you could plausibly deploy different strategies in?

Nothing here is a trading signal. The GMM is fitted on the full sample, so the
labels use information from the end of the sample to describe the beginning.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Return series
# ---------------------------------------------------------------------------

def prepare_returns(df: pd.DataFrame, return_col: str) -> dict:
    """Return a gap-free daily return series plus a record of what was filled.

    The per-expiry return columns carry a NaN on every roll date, because the
    first day of a new contract has no prior close to difference against. Those
    NaNs are sparse but evenly spread — the longest clean run in this series is
    ~145 days — so any rolling window wider than that is NaN *everywhere* and a
    later dropna() silently empties the frame.

    Filling roll gaps with zero is the least distorting assumption for rolling
    vol and drift. It is not a claim that the market was flat on those days; it
    keeps ~1% of observations from destroying every long-window feature.
    """
    raw = df[return_col]
    gaps = raw.isna()
    filled = raw.fillna(0.0)
    longest_clean = int(np.diff(np.r_[-1, np.where(gaps.to_numpy())[0], len(raw)]).max())

    return {
        "returns": filled,
        "gap_mask": gaps,
        "n_gaps": int(gaps.sum()),
        "n_obs": int(len(raw)),
        "gap_fraction": float(gaps.mean()),
        "longest_clean_run": longest_clean,
        "gap_dates": raw.index[gaps],
    }


# ---------------------------------------------------------------------------
# 2. Regime feature space
# ---------------------------------------------------------------------------

DEFAULT_FEATURES = (
    "log_vol",
    "vol_ratio",
    "trend_z",
    "var_ratio",
    "arb_pctl",
    "cbot_corr",
)


def build_regime_features(
    df: pd.DataFrame,
    return_col: str = "ret_safex_zar_per_tonne",
    usd_return_col: str = "ret_safex_usd_per_tonne",
    cbot_return_col: str = "ret_cbot_usd_per_tonne",
    arb_col: str = "ym_corn_arb",
    oi_col: str = "safex_open_interest",
    vol_short: int = 21,
    vol_long: int = 63,
    var_ratio_lag: int = 5,
    pctl_window: int = 756,
    include: tuple[str, ...] = DEFAULT_FEATURES,
    rolling_z_vol: bool = False,
) -> pd.DataFrame:
    """Backward-looking descriptors of the state of the SAFEX maize market.

    Two families, deliberately mixed. Price-statistical features describe the
    *shape* of the return process; structural features describe *where the
    local market sits relative to world parity*, which for SAFEX maize is the
    thing that actually determines what trades work.

      log_vol     how violent is it. Logged because realised vol is roughly
                  lognormal, and GMM components are Gaussian — fitting raw vol
                  forces the model to spend a component on the right tail.
      vol_ratio   log(21d vol / 63d vol). Vol accelerating or decaying, which
                  separates "calm", "blowing up" and "cooling off" — three
                  states that share a vol level but want opposite gross.
      trend_z     63d drift in units of its own noise. Directional persistence.
      var_ratio   63d variance ratio at lag 5. Above 1 moves compound, below 1
                  they offset. The most direct trend-vs-fade discriminator.
      arb_pctl    rolling percentile of the SAFEX-CBOT basis. This is the
                  import/export parity band position. Near the top the local
                  price is pinned to import parity (shortage, tethered to CBOT
                  plus freight plus FX); near the bottom it is at export parity
                  (surplus, pinned to the floor); mid-band it floats on local
                  supply and demand. Completely different trades in each.
      cbot_corr   63d correlation of SAFEX USD returns to CBOT USD returns.
                  Whether the market is globally or locally driven — and
                  therefore whether a CBOT hedge works at all. Deliberately in
                  USD on both legs so FX noise does not contaminate it.
      oi_trend    optional. 21d change in log open interest, z-scored. Proxy
                  for commercial hedging pressure building or unwinding.
      drawdown_z  optional. Depth below the 126d high in vol units. Separates
                  "calm and grinding up" from "calm and bleeding".

    Seasonality is deliberately NOT a feature. Feeding month or season_sin into
    the GMM would force the clusters to become calendar buckets and destroy the
    only interesting question — whether the discovered regimes line up with the
    calendar on their own. Season is the validation axis, not an input.

    No forward information is used; every value at t is computable at t. Roll
    gaps in every return input are zero-filled first, see `prepare_returns`.
    """
    returns = df[return_col].fillna(0.0)
    cum = returns.cumsum()

    vol_s = returns.rolling(vol_short).std()
    vol_l = returns.rolling(vol_long).std()

    log_vol = np.log(vol_l * np.sqrt(TRADING_DAYS))
    if rolling_z_vol:
        # Removes slow drift in the overall vol level, at the cost of making a
        # violent era and a calm one look alike. Off by default: the absolute
        # level is what you size against. Use temporal_split_stability to find
        # out whether leaving it in makes the buckets era-dependent.
        log_vol = (log_vol - log_vol.rolling(TRADING_DAYS).mean()) / log_vol.rolling(
            TRADING_DAYS
        ).std()

    var_1 = returns.rolling(vol_long).var()
    var_k = cum.diff(var_ratio_lag).rolling(vol_long).var()

    usd_ret = df[usd_return_col].fillna(0.0)
    cbot_ret = df[cbot_return_col].fillna(0.0)

    arb = df[arb_col]
    arb_pctl = arb.rolling(pctl_window, min_periods=pctl_window // 3).rank(pct=True)

    log_oi = np.log(df[oi_col].replace(0, np.nan))
    oi_change = log_oi.diff(vol_short)

    running_max = cum.rolling(126).max()

    candidates = {
        "log_vol": log_vol,
        "vol_ratio": np.log(vol_s / vol_l),
        "trend_z": returns.rolling(vol_long).sum() / (vol_l * np.sqrt(vol_long)),
        "var_ratio": var_k / (var_ratio_lag * var_1),
        "arb_pctl": arb_pctl,
        "cbot_corr": usd_ret.rolling(vol_long).corr(cbot_ret),
        "oi_trend": (oi_change - oi_change.rolling(TRADING_DAYS).mean())
        / oi_change.rolling(TRADING_DAYS).std(),
        "drawdown_z": (cum - running_max) / (vol_l * np.sqrt(vol_long)),
    }

    unknown = set(include) - set(candidates)
    if unknown:
        raise ValueError(f"unknown regime features: {sorted(unknown)}")

    features = pd.DataFrame({name: candidates[name] for name in include}, index=df.index)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def feature_diagnostics(features: pd.DataFrame) -> dict:
    """Correlation and multicollinearity check on the feature space.

    A GMM with full covariance on features correlated at 0.9 is fitting a
    degenerate cigar and will split it arbitrarily along the long axis. If two
    features carry a |rho| above ~0.8, drop one before believing any bucketing.
    VIF above ~5 says the same thing with more decimal places.
    """
    corr = features.corr().round(3)
    scaled = StandardScaler().fit_transform(features)
    inv = np.linalg.pinv(np.corrcoef(scaled, rowvar=False))
    vif = pd.Series(np.diag(inv), index=features.columns, name="vif").round(2)

    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    worst = off_diag.abs().stack()
    worst_pair = worst.idxmax() if len(worst) else None

    return {
        "correlation": corr,
        "vif": vif,
        "max_abs_correlation": float(worst.max()) if len(worst) else np.nan,
        "worst_pair": worst_pair,
        "max_vif": float(vif.max()),
    }


def effective_sample_size(features: pd.DataFrame, max_lag: int = 500) -> dict:
    """How many *independent* observations the feature matrix really contains.

    Rolling features are heavily autocorrelated: 3,900 daily rows built from
    63-day windows are nowhere near 3,900 independent draws. BIC, silhouette
    and every p-value downstream assume iid observations, so they are all
    flattered by the raw row count. This estimates the integrated
    autocorrelation time per feature and divides through.

    Treat the result as an order of magnitude, not a precise number.
    """
    taus = {}
    for col in features.columns:
        x = features[col].to_numpy(dtype=float)
        x = x - x.mean()
        n = len(x)
        denom = np.dot(x, x)
        if denom == 0:
            taus[col] = 1.0
            continue
        rho_sum = 0.0
        for lag in range(1, min(max_lag, n - 1)):
            rho = np.dot(x[:-lag], x[lag:]) / denom
            if rho <= 0:                      # initial-positive-sequence cutoff
                break
            rho_sum += rho
        taus[col] = 1.0 + 2.0 * rho_sum

    tau_mean = float(np.mean(list(taus.values())))
    n_rows = int(len(features))
    return {
        "autocorr_time": taus,
        "mean_autocorr_time": tau_mean,
        "n_rows": n_rows,
        "n_effective": float(n_rows / tau_mean),
    }


# ---------------------------------------------------------------------------
# 3. Model selection
# ---------------------------------------------------------------------------

def _n_params(k: int, d: int, covariance_type: str) -> int:
    mean_p = k * d
    weight_p = k - 1
    cov_p = {
        "full": k * d * (d + 1) // 2,
        "tied": d * (d + 1) // 2,
        "diag": k * d,
        "spherical": k,
    }[covariance_type]
    return mean_p + weight_p + cov_p


def select_n_components(
    features: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    covariance_types: tuple[str, ...] = ("full", "diag", "tied", "spherical"),
    n_init: int = 10,
    seed: int = 0,
    n_effective: float | None = None,
) -> dict:
    """Fit the grid and score it, with an autocorrelation-adjusted BIC.

    `bic_eff` rescales the log-likelihood and the penalty to the effective
    sample size. Raw BIC on autocorrelated daily data will happily justify eight
    components; bic_eff is the honest version and usually points much lower.
    """
    scaled = StandardScaler().fit_transform(features)
    n, d = scaled.shape
    rows = []

    for cov in covariance_types:
        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                random_state=seed,
                n_init=n_init,
                reg_covar=1e-5,
            ).fit(scaled)

            loglik = float(gmm.score(scaled) * n)
            p = _n_params(k, d, cov)
            row = {
                "k": k,
                "covariance_type": cov,
                "loglik": loglik,
                "n_params": p,
                "bic": float(gmm.bic(scaled)),
                "aic": float(gmm.aic(scaled)),
                "converged": bool(gmm.converged_),
            }
            if n_effective is not None:
                row["bic_eff"] = float(
                    -2.0 * loglik * (n_effective / n) + p * np.log(n_effective)
                )
            if k > 1:
                labels = gmm.predict(scaled)
                row["silhouette"] = (
                    float(silhouette_score(scaled, labels))
                    if len(np.unique(labels)) > 1
                    else np.nan
                )
                row["min_weight"] = float(gmm.weights_.min())
            rows.append(row)

    table = pd.DataFrame(rows)
    criterion = "bic_eff" if n_effective is not None else "bic"
    best = table.loc[table[criterion].idxmin()]

    return {
        "table": table,
        "criterion": criterion,
        "best_k": int(best["k"]),
        "best_covariance_type": str(best["covariance_type"]),
        "scaler_fitted_on": list(features.columns),
    }


# ---------------------------------------------------------------------------
# 4. Fit and canonical labelling
# ---------------------------------------------------------------------------

def fit_regimes(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    seed: int = 0,
    n_init: int = 20,
    order_by: str = "log_vol",
) -> dict:
    """Fit the GMM and relabel components in ascending order of `order_by`.

    GMM component numbering is arbitrary and changes with the seed. Sorting by
    mean volatility makes regime 0 the calmest every time, so labels are
    comparable across refits, across feature sets and across conversations.
    """
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        n_init=n_init,
        reg_covar=1e-5,
    ).fit(scaled)

    raw_labels = gmm.predict(scaled)
    posteriors = gmm.predict_proba(scaled)

    centres = pd.DataFrame(
        scaler.inverse_transform(gmm.means_), columns=features.columns
    )
    order = centres[order_by].sort_values().index.to_numpy()
    remap = {int(old): int(new) for new, old in enumerate(order)}

    labels = pd.Series(
        [remap[int(v)] for v in raw_labels], index=features.index, name="regime"
    )
    posteriors = pd.DataFrame(
        posteriors[:, order],
        index=features.index,
        columns=[f"p_regime_{i}" for i in range(n_components)],
    )
    centres = centres.iloc[order].reset_index(drop=True)
    centres.index.name = "regime"

    return {
        "model": gmm,
        "scaler": scaler,
        "labels": labels,
        "posteriors": posteriors,
        "centres": centres,
        "weights": pd.Series(gmm.weights_[order], name="weight"),
        "label_map": remap,
        "converged": bool(gmm.converged_),
        "n_components": n_components,
        "covariance_type": covariance_type,
    }


# ---------------------------------------------------------------------------
# 5. Reliability diagnostics — the actual question being asked
# ---------------------------------------------------------------------------

def assignment_confidence(posteriors: pd.DataFrame, threshold: float = 0.80) -> dict:
    """How decisive the bucketing is, day by day.

    A GMM always returns a label. If the top posterior sits near 1/k most days,
    the components overlap and the "regimes" are slices of one blob.
    """
    top = posteriors.max(axis=1)
    k = posteriors.shape[1]
    p = posteriors.to_numpy()
    entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(k)

    return {
        "max_posterior": top,
        "mean_max_posterior": float(top.mean()),
        "median_max_posterior": float(top.median()),
        "share_confident": float((top >= threshold).mean()),
        "threshold": threshold,
        "uniform_baseline": 1.0 / k,
        "mean_normalised_entropy": float(entropy.mean()),
        "entropy": pd.Series(entropy, index=posteriors.index, name="entropy"),
    }


def regime_persistence(labels: pd.Series) -> dict:
    """Run lengths and switch rate.

    A bucket you occupy for three days at a time is not a regime you can deploy
    a strategy in, however clean the clustering looks. This is the diagnostic
    that most often kills an otherwise attractive fit.
    """
    values = labels.to_numpy()
    change = np.r_[True, values[1:] != values[:-1]]
    run_id = np.cumsum(change)
    runs = pd.DataFrame({"regime": values, "run_id": run_id})
    run_lengths = runs.groupby(["run_id", "regime"]).size().reset_index(name="length")

    per_regime = (
        run_lengths.groupby("regime")["length"]
        .agg(n_spells="count", mean_length="mean", median_length="median", max_length="max")
        .round(2)
    )
    per_regime["occupancy"] = labels.value_counts(normalize=True).sort_index().round(4)

    return {
        "per_regime": per_regime,
        "run_lengths": run_lengths,
        "switch_rate": float(change[1:].mean()),
        "n_spells": int(run_lengths.shape[0]),
        "overall_median_length": float(run_lengths["length"].median()),
    }


def label_stability(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    n_seeds: int = 15,
    n_block_samples: int = 25,
    block_size: int = 63,
    sample_fraction: float = 0.8,
    base_seed: int = 0,
) -> dict:
    """Do you get the same buckets back if you perturb the fit?

    Two perturbations, both reported as adjusted Rand index against a reference
    fit. ARI is permutation-invariant, so component renumbering does not count
    as instability. 1.0 is identical, 0.0 is chance.

      seed_ari    refit the same data from different random starts. Low values
                  mean the likelihood surface is flat and the fit is arbitrary.
      block_ari   refit on contiguous block subsamples, then label the full
                  sample. Blocks rather than iid rows because the rows are
                  autocorrelated and iid resampling would leak neighbours into
                  every draw and overstate stability badly.
    """
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    n = len(scaled)

    def _fit(data, seed):
        return GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=seed,
            n_init=10,
            reg_covar=1e-5,
        ).fit(data)

    reference = _fit(scaled, base_seed).predict(scaled)

    seed_ari = [
        adjusted_rand_score(reference, _fit(scaled, base_seed + s).predict(scaled))
        for s in range(1, n_seeds + 1)
    ]

    rng = np.random.default_rng(base_seed)
    n_blocks_total = max(1, n // block_size)
    n_blocks_draw = max(1, int(round(n_blocks_total * sample_fraction)))
    block_ari = []
    for _ in range(n_block_samples):
        starts = rng.choice(n_blocks_total, size=n_blocks_draw, replace=False)
        idx = np.concatenate(
            [np.arange(s * block_size, min((s + 1) * block_size, n)) for s in starts]
        )
        try:
            model = _fit(scaled[idx], base_seed)
            block_ari.append(adjusted_rand_score(reference, model.predict(scaled)))
        except ValueError:
            continue

    seed_ari = np.array(seed_ari)
    block_ari = np.array(block_ari)

    return {
        "seed_ari": seed_ari,
        "block_ari": block_ari,
        "seed_ari_mean": float(seed_ari.mean()),
        "seed_ari_min": float(seed_ari.min()),
        "block_ari_mean": float(block_ari.mean()) if block_ari.size else np.nan,
        "block_ari_median": float(np.median(block_ari)) if block_ari.size else np.nan,
        "block_ari_p10": float(np.percentile(block_ari, 10)) if block_ari.size else np.nan,
        "block_size": block_size,
        "n_components": n_components,
    }


def cross_algorithm_agreement(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    seed: int = 0,
) -> dict:
    """Do other clustering geometries find the same buckets?

    GMM assumes elliptical Gaussian components. K-means assumes spherical ones
    of similar size; Ward assumes compact, variance-minimising ones. If all
    three land on roughly the same partition, the structure is in the data. If
    only the GMM sees it, what you are looking at is the Gaussian assumption
    talking, and the buckets will not survive contact with new data.

    Rule of thumb: ARI above ~0.5 across algorithms is real structure, below
    ~0.3 is one blob being sliced three different ways.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans

    scaled = StandardScaler().fit_transform(features)

    gmm_labels = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        n_init=10,
        reg_covar=1e-5,
    ).fit_predict(scaled)
    km_labels = KMeans(n_clusters=n_components, random_state=seed, n_init=10).fit_predict(scaled)
    ward_labels = AgglomerativeClustering(n_clusters=n_components).fit_predict(scaled)

    pairs = {
        "gmm_vs_kmeans": adjusted_rand_score(gmm_labels, km_labels),
        "gmm_vs_ward": adjusted_rand_score(gmm_labels, ward_labels),
        "kmeans_vs_ward": adjusted_rand_score(km_labels, ward_labels),
    }
    return {
        "pairwise_ari": pd.Series(pairs).round(3),
        "min_ari": float(min(pairs.values())),
        "mean_ari": float(np.mean(list(pairs.values()))),
        "labels": {"gmm": gmm_labels, "kmeans": km_labels, "ward": ward_labels},
    }


def temporal_split_stability(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    n_folds: int = 3,
    seed: int = 0,
    order_by: str = "log_vol",
) -> dict:
    """Is the regime *structure* the same in different eras of the sample?

    Split the history into contiguous folds, fit each separately, and compare
    the vol-ordered centroids. If the 2010-2015 "high vol" bucket sits at a
    different place in feature space from the 2021-2026 one, you have not found
    stable regimes, you have found a drifting distribution that the GMM is
    re-slicing each time.
    """
    folds = np.array_split(np.arange(len(features)), n_folds)
    centre_frames = {}
    for i, idx in enumerate(folds):
        sub = features.iloc[idx]
        fit = fit_regimes(
            sub, n_components, covariance_type, seed=seed, n_init=10, order_by=order_by
        )
        centres = fit["centres"].copy()
        centres["fold"] = i
        centres["start"] = sub.index[0]
        centres["end"] = sub.index[-1]
        centre_frames[i] = centres

    stacked = pd.concat(centre_frames.values()).reset_index()

    # Spread of each regime's centroid across folds, in units of the feature's
    # own full-sample standard deviation. Above ~0.5 is a meaningful drift.
    scale = features.std()
    drift = (
        stacked.groupby("regime")[list(features.columns)].std() / scale
    ).round(3)

    return {
        "centres_by_fold": stacked,
        "centroid_drift": drift,
        "max_drift": float(drift.max().max()),
        "n_folds": n_folds,
    }


def run_reliability_battery(
    features: pd.DataFrame,
    k_values: tuple[int, ...] = (2, 3, 4, 5),
    covariance_type: str = "full",
    selection_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every reliability diagnostic, one row per candidate component count."""
    rows = []
    for k in k_values:
        fit = fit_regimes(features, k, covariance_type, n_init=20)
        conf = assignment_confidence(fit["posteriors"])
        pers = regime_persistence(fit["labels"])
        stab = label_stability(features, k, covariance_type, n_seeds=8, n_block_samples=15)
        xalg = cross_algorithm_agreement(features, k, covariance_type)

        row = {
            "k": k,
            "share_confident": conf["share_confident"],
            "seed_ARI": stab["seed_ari_mean"],
            "block_ARI": stab["block_ari_median"],
            "xalgo_ARI": xalg["min_ari"],
            "median_run_d": pers["overall_median_length"],
            "mean_run_d": pers["run_lengths"]["length"].mean(),
            "switch_rate": pers["switch_rate"],
        }
        if selection_table is not None:
            match = selection_table[
                (selection_table["k"] == k)
                & (selection_table["covariance_type"] == covariance_type)
            ]
            row["silhouette"] = float(match["silhouette"].iloc[0]) if len(match) else np.nan
        rows.append(row)

    columns = ["k", "silhouette", "share_confident", "seed_ARI", "block_ARI",
               "xalgo_ARI", "median_run_d", "mean_run_d", "switch_rate"]
    table = pd.DataFrame(rows).set_index("k")
    return table[[c for c in columns if c in table.columns]].round(3)


SWEEP_CONFIGS: dict[str, dict] = {
    "A  6f / 63d": dict(include=DEFAULT_FEATURES, vol_long=63, vol_short=21, pctl_window=756),
    "B  3f / 63d": dict(include=("log_vol", "var_ratio", "arb_pctl"), vol_long=63,
                        vol_short=21, pctl_window=756),
    "C  3f / 21d": dict(include=("log_vol", "var_ratio", "arb_pctl"), vol_long=21,
                        vol_short=10, pctl_window=252),
    "D  vol+arb / 21d": dict(include=("log_vol", "arb_pctl"), vol_long=21,
                             vol_short=10, pctl_window=252),
    "E  price-only 4f": dict(include=("log_vol", "vol_ratio", "trend_z", "var_ratio"),
                             vol_long=42, vol_short=15),
}


def run_config_sweep(
    df: pd.DataFrame,
    configs: dict[str, dict] | None = None,
    k_values: tuple[int, ...] = (2, 3, 4),
) -> pd.DataFrame:
    """Does any feature configuration produce buckets that hold together?

    Shorter windows buy back effective sample size; fewer dimensions make
    components easier to separate. If nothing here clears the pass marks, the
    problem is the data rather than the choice of feature set.
    """
    configs = configs or SWEEP_CONFIGS
    rows = []
    for name, kwargs in configs.items():
        features = build_regime_features(df, **kwargs)
        ess = effective_sample_size(features)
        selection = select_n_components(
            features, k_values=(1,) + k_values, n_effective=ess["n_effective"]
        )
        for k in k_values:
            stab = label_stability(features, k, "full", n_seeds=6, n_block_samples=12)
            xalg = cross_algorithm_agreement(features, k, "full")
            pers = regime_persistence(fit_regimes(features, k, "full", n_init=15)["labels"])
            match = selection["table"].query("k == @k and covariance_type == 'full'")
            rows.append({
                "config": name,
                "d": features.shape[1],
                "ESS": round(ess["n_effective"], 1),
                "bic_eff_k": selection["best_k"],
                "k": k,
                "silhouette": float(match["silhouette"].iloc[0]),
                "block_ARI": stab["block_ari_median"],
                "xalgo_ARI": xalg["min_ari"],
                "median_run_d": pers["overall_median_length"],
                "mean_run_d": pers["run_lengths"]["length"].mean(),
            })
    return pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------------------
# 6. Profiling — what would you actually trade in each bucket
# ---------------------------------------------------------------------------

def profile_regimes(
    labels: pd.Series,
    returns: pd.Series,
    features: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """Descriptive statistics per bucket, contemporaneous and forward.

    IMPORTANT: the labels come from a full-sample fit, so the forward-return
    columns are *not* an edge. They describe what the series did while it was in
    a state you can only identify with hindsight. Their job here is to tell you
    whether the buckets differ in a way worth building separate strategies for —
    if every bucket has the same vol and the same mean reversion, the split is
    not useful regardless of how clean the clustering was.
    """
    aligned = returns.reindex(labels.index)
    frame = pd.DataFrame({"regime": labels, "ret": aligned})

    for h in horizons:
        fwd = returns.rolling(h).sum().shift(-h + 1) if h > 1 else returns
        frame[f"fwd_{h}d"] = fwd.reindex(labels.index)

    def _stats(group: pd.DataFrame) -> pd.Series:
        r = group["ret"]
        cum = r.cumsum()
        out = {
            "n_days": len(r),
            "share": len(r) / len(frame),
            "ann_return": r.mean() * TRADING_DAYS,
            "ann_vol": r.std() * np.sqrt(TRADING_DAYS),
            "sharpe": (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
            "hit_rate": (r > 0).mean(),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
            "max_drawdown": float((np.exp(cum) / np.exp(cum).cummax() - 1).min()),
        }
        for h in horizons:
            col = group[f"fwd_{h}d"].dropna()
            out[f"fwd_{h}d_mean"] = col.mean()
            out[f"fwd_{h}d_t"] = (
                col.mean() / col.std() * np.sqrt(len(col)) if col.std() > 0 else np.nan
            )
        return pd.Series(out)

    summary = frame.groupby("regime", group_keys=False).apply(_stats, include_groups=False).round(4)
    feature_means = features.groupby(labels).mean().round(4)
    feature_means.index.name = "regime"

    month = pd.Series(labels.index.month, index=labels.index, name="month")
    seasonality = pd.crosstab(month, labels, normalize="index").round(3)
    seasonality_counts = pd.crosstab(month, labels)

    year = pd.Series(labels.index.year, index=labels.index, name="year")
    by_year = pd.crosstab(year, labels, normalize="index").round(3)

    return {
        "summary": summary,
        "feature_means": feature_means,
        "seasonality": seasonality,
        "seasonality_counts": seasonality_counts,
        "occupancy_by_year": by_year,
        "panel": frame,
    }


# ---------------------------------------------------------------------------
# 7. Optional: the honest out-of-sample version
# ---------------------------------------------------------------------------

def walk_forward_labels(
    features: pd.DataFrame,
    n_components: int,
    covariance_type: str = "full",
    lookback: int = 756,
    refit_every: int = 21,
    seed: int = 0,
    order_by: str = "log_vol",
) -> dict:
    """Label each day using only data available before it.

    Not needed for a descriptive study, but it is the only version whose labels
    you may feed into a backtest. Refits monthly rather than daily — a daily
    refit is ~3,000 GMM fits for no statistical gain, and it was the shape of
    the loop that made the earlier attempt unusably slow.

    Components are vol-ordered at every refit, so labels stay comparable across
    refits instead of permuting randomly.
    """
    n = len(features)
    labels = pd.Series(np.nan, index=features.index, name="regime_oos")
    confidence = pd.Series(np.nan, index=features.index, name="max_posterior")
    refit_dates = []

    if lookback >= n:
        raise ValueError(
            f"lookback={lookback} exceeds available rows ({n}); "
            "shorten the lookback or the feature windows."
        )

    for start in range(lookback, n, refit_every):
        train = features.iloc[start - lookback : start]
        test = features.iloc[start : start + refit_every]
        if test.empty:
            break

        scaler = StandardScaler().fit(train)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=seed,
            n_init=5,
            reg_covar=1e-5,
        ).fit(scaler.transform(train))

        centres = pd.DataFrame(
            scaler.inverse_transform(gmm.means_), columns=features.columns
        )
        order = centres[order_by].sort_values().index.to_numpy()
        remap = {int(old): int(new) for new, old in enumerate(order)}

        scaled_test = scaler.transform(test)
        post = gmm.predict_proba(scaled_test)
        labels.iloc[start : start + len(test)] = [
            remap[int(v)] for v in gmm.predict(scaled_test)
        ]
        confidence.iloc[start : start + len(test)] = post.max(axis=1)
        refit_dates.append(features.index[start])

    valid = labels.dropna()
    return {
        "labels": labels.dropna().astype(int),
        "confidence": confidence.dropna(),
        "n_labelled": int(len(valid)),
        "n_refits": len(refit_dates),
        "refit_dates": pd.DatetimeIndex(refit_dates),
        "first_labelled": valid.index[0] if len(valid) else None,
    }


# ---------------------------------------------------------------------------
# 8. How the arb behaves in each state
# ---------------------------------------------------------------------------

def prepare_arb(
    df: pd.DataFrame,
    arb_col: str = "ym_corn_arb",
    return_col: str = "ret_safex_zar_per_tonne",
    ym_col: str = "ret_safex_usd_per_tonne",
    cbot_col: str = "ret_cbot_usd_per_tonne",
) -> dict:
    """Clean daily change in the arb, plus both legs, on a common roll mask.

    Both legs jump on a roll date, so differencing the arb level across one
    measures the roll, not the market. Those days are zeroed rather than left as
    NaN — a NaN inside a 20-day forward sum voids the whole window, which
    silently deletes the months that follow a roll and biases any month-level
    table built on top.
    """
    gaps = df[return_col].isna()
    return {
        "arb_level": df[arb_col],
        "arb_change": df[arb_col].diff().mask(gaps).fillna(0.0),
        "ym": df[ym_col].mask(gaps).fillna(0.0),
        "cbot": df[cbot_col].mask(gaps).fillna(0.0),
        "gap_mask": gaps,
    }


def arb_response(
    group_key: pd.Series,
    arb: dict,
    horizon: int = 20,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """Two-sided read on the arb for each value of `group_key`.

    `group_key` is whatever you want to condition on — a regime label, a
    calendar month, a quintile. Both directions are reported, because the book
    trades both:

      buying the arb  = long SAFEX, short CBOT. Wins when the arb widens.
      selling the arb = short SAFEX, long CBOT. Wins when the arb narrows.

    So `fwd_mean` above zero favours the long side and hurts the short side.
    `p90`/`pct_widen` size the pain for a seller; `p10`/`pct_narrow` size it for
    a buyer. The t-statistic is divided by sqrt(horizon) because overlapping
    forward windows are not independent observations — treat |t_adj| under 2 as
    a tendency worth sizing around, not a signal worth entering on.
    """
    fwd = arb["arb_change"].rolling(horizon).sum().shift(-horizon + 1)

    panel = pd.DataFrame(
        {
            "key": group_key,
            "fwd": fwd.reindex(group_key.index),
            "level": arb["arb_level"].reindex(group_key.index),
            "chg": arb["arb_change"].reindex(group_key.index),
            "ym": arb["ym"].reindex(group_key.index),
            "cbot": arb["cbot"].reindex(group_key.index),
        }
    ).dropna(subset=["fwd"])

    def _stats(g: pd.DataFrame) -> pd.Series:
        f = g["fwd"]
        return pd.Series(
            {
                "n": len(f),
                "arb_level": g["level"].mean(),
                "ym_ann_pct": g["ym"].mean() * TRADING_DAYS * 100,
                "cbot_ann_pct": g["cbot"].mean() * TRADING_DAYS * 100,
                "corr_arb_ym": g["chg"].corr(g["ym"]),
                "corr_arb_cbot": g["chg"].corr(g["cbot"]),
                "corr_ym_cbot": g["ym"].corr(g["cbot"]),
                "fwd_mean": f.mean(),
                "fwd_sd": f.std(),
                "t_adj": f.mean() / f.std() * np.sqrt(len(f)) / np.sqrt(horizon)
                if f.std() > 0
                else np.nan,
                "p10_narrow": f.quantile(0.10),
                "p90_widen": f.quantile(0.90),
                "pct_widen": (f > threshold).mean() * 100,
                "pct_narrow": (f < -threshold).mean() * 100,
                "worst_widen": f.max(),
                "worst_narrow": f.min(),
            }
        )

    return panel.groupby("key", observed=True).apply(_stats, include_groups=False)


def arb_level_quintiles(
    arb: dict,
    index: pd.Index,
    rank_window: int = 756,
    horizon: int = 20,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Does a wide arb come back, or keep going?

    Sorts every day by where the arb sat in its own trailing distribution — a
    point-in-time rank, so no future information — and reports what the arb did
    next. A downward-sloping column means the arb mean-reverts and fading the
    extreme is right. An upward-sloping one means it trends, and fading it is
    backwards.

    This is the non-circular version of the same question. Clustering on the
    arb level and then measuring the arb change inside each cluster answers
    itself; ranking and looking forward does not.
    """
    rank = (
        arb["arb_level"]
        .rolling(rank_window, min_periods=rank_window // 3)
        .rank(pct=True)
        .reindex(index)
        .dropna()
    )
    labels = [f"Q{i}" for i in range(1, n_buckets + 1)]
    buckets = pd.qcut(rank, n_buckets, labels=labels)
    return arb_response(buckets, arb, horizon=horizon)


def compare_insample_oos(insample: pd.Series, oos: pd.Series) -> dict:
    """Agreement between the full-sample labels and the walk-forward labels.

    The gap between these two is the size of the hindsight premium in the
    descriptive study.
    """
    common = insample.index.intersection(oos.index)
    a = insample.reindex(common)
    b = oos.reindex(common)
    return {
        "n_common": int(len(common)),
        "agreement": float((a == b).mean()),
        "adjusted_rand": float(adjusted_rand_score(a, b)),
        "confusion": pd.crosstab(a.rename("in_sample"), b.rename("walk_forward")),
    }