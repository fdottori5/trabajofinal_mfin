
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# =========================
# 1) INPUT / PREPROCESSING
# =========================

def load_bloomberg_dataset(
    input_path: str | Path,
    sheet_name: str | int = 0,
    date_col: str = "FECHA",
) -> pd.DataFrame:
    """
    Lee una bajada de Bloomberg en CSV o Excel.

    Espera una columna de fecha y luego columnas numéricas.
    Elimina columnas espurias tipo 'Unnamed: 0'.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(
            input_path,
            sheet_name=sheet_name,
            na_values=["#N/A", "#N/A N/A", "N/A", "NA"],
        )
    elif suffix == ".csv":
        df = pd.read_csv(
            input_path,
            na_values=["#N/A", "#N/A N/A", "N/A", "NA"],
        )
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usá CSV o Excel.")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    if date_col not in df.columns:
        raise ValueError(f"No encontré la columna de fecha '{date_col}'. Columnas: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    value_cols = [c for c in df.columns if c != date_col]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.drop_duplicates(subset=[date_col], keep="last")
          .sort_values(date_col)
          .set_index(date_col)
    )

    return df[value_cols]


def prepare_price_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    prices = df[columns].copy()
    prices = prices.ffill().dropna(how="any")
    if prices.empty:
        raise ValueError(f"No quedaron precios válidos para las columnas: {columns}")
    return prices


def daily_prices_to_monthly_returns(prices_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly_prices = prices_daily.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="any")
    return monthly_prices, monthly_returns


def daily_yield_to_monthly_rf_returns(yield_daily: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Convierte una serie diaria de yields anuales (%) a retornos mensuales decimales.
    """
    y = yield_daily.copy().ffill().dropna()
    if y.empty:
        raise ValueError("La serie de tasa libre de riesgo quedó vacía.")

    monthly_yield_pct = y.resample("ME").last()
    monthly_yield_decimal = monthly_yield_pct / 100.0
    monthly_rf_same_month = (1.0 + monthly_yield_decimal) ** (1.0 / 12.0) - 1.0
    monthly_rf_for_next_period = monthly_rf_same_month.shift(1)
    return monthly_yield_pct, monthly_rf_for_next_period


def align_monthly_inputs(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rf_monthly: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    common_index = asset_returns.index.intersection(benchmark_returns.index).intersection(rf_monthly.index)
    asset_returns = asset_returns.loc[common_index].copy()
    benchmark_returns = benchmark_returns.loc[common_index].copy()
    rf_monthly = rf_monthly.loc[common_index].copy()

    valid_mask = (~asset_returns.isna().any(axis=1)) & benchmark_returns.notna() & rf_monthly.notna()
    asset_returns = asset_returns.loc[valid_mask]
    benchmark_returns = benchmark_returns.loc[valid_mask]
    rf_monthly = rf_monthly.loc[valid_mask]

    if asset_returns.empty:
        raise ValueError("No quedaron observaciones mensuales comunes entre activos, benchmark y RF.")

    return asset_returns, benchmark_returns, rf_monthly


# =========================
# 2) ESTIMACIÓN DE INPUTS
# =========================

def annualize_mean(monthly_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return monthly_returns.mean() * 12.0


def annualize_cov(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    return monthly_returns.cov() * 12.0


def estimate_historical_mu_cov(returns_monthly: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu = annualize_mean(returns_monthly)
    cov = annualize_cov(returns_monthly)
    return mu, cov


def estimate_capm_expected_returns(
    asset_returns_monthly: pd.DataFrame,
    benchmark_returns_monthly: pd.Series,
    rf_monthly: pd.Series,
) -> Tuple[pd.Series, pd.Series, float, float]:
    asset_excess = asset_returns_monthly.sub(rf_monthly, axis=0)
    market_excess = benchmark_returns_monthly - rf_monthly

    market_var = float(market_excess.var(ddof=1))
    if market_var <= 0:
        raise ValueError("La varianza del benchmark en exceso es no positiva; no se puede estimar CAPM.")

    betas = {}
    for col in asset_excess.columns:
        cov_im = float(np.cov(asset_excess[col].values, market_excess.values, ddof=1)[0, 1])
        betas[col] = cov_im / market_var
    betas = pd.Series(betas, name="beta_capm")

    rf_annual = float(rf_monthly.mean() * 12.0)
    market_premium_annual = float(market_excess.mean() * 12.0)
    mu_capm_annual = rf_annual + betas * market_premium_annual
    mu_capm_annual.name = "expected_return_capm_annual"

    return betas, mu_capm_annual, rf_annual, market_premium_annual


# =========================
# 3) MODELOS BASE
# =========================

def _normalize_long_only(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("No se pudieron normalizar los pesos: suma no positiva.")
    return weights / total


def get_upper_bounds(asset_index: pd.Index | List[str]) -> pd.Series:
    ub = pd.Series(1.0, index=list(asset_index), dtype=float)
    if "BTC" in ub.index:
        ub["BTC"] = 0.05
    if "GLD" in ub.index:
        ub["GLD"] = 0.15
    return ub


def project_with_caps(weights: pd.Series, upper_bounds: pd.Series, name: str = "caps_projection") -> pd.Series:
    idx = list(weights.index)
    ub = upper_bounds.reindex(idx).fillna(1.0).astype(float)
    x0 = np.clip(weights.values, 0.0, ub.values)

    def objective(w: np.ndarray) -> float:
        return float(np.sum((w - weights.values) ** 2))

    bounds = [(0.0, float(ub[a])) for a in idx]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Falló {name}: {result.message}")

    return pd.Series(result.x, index=idx, name=weights.name)


def maximum_sharpe_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    name: str = "maximum_sharpe",
    upper_bounds: pd.Series | None = None,
) -> pd.Series:
    n = len(mu)
    mu_np = mu.values
    cov_np = cov.values

    if upper_bounds is None:
        upper_bounds = pd.Series(1.0, index=mu.index, dtype=float)

    def objective(w: np.ndarray) -> float:
        port_ret = float(w @ mu_np)
        port_var = float(w @ cov_np @ w)
        if port_var <= 0:
            return 1e6
        port_vol = float(np.sqrt(port_var))
        sharpe = (port_ret - rf) / port_vol
        return -sharpe

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, float(upper_bounds[a])) for a in mu.index]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Falló maximum_sharpe_weights ({name}): {result.message}")

    weights = _normalize_long_only(result.x)
    return pd.Series(weights, index=mu.index, name=name)


def _risk_contributions(weights: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    portfolio_vol = float(np.sqrt(np.maximum(weights @ cov_np @ weights, 0.0)))
    if portfolio_vol <= 0:
        return np.zeros_like(weights)
    marginal_risk = cov_np @ weights / portfolio_vol
    return weights * marginal_risk


def risk_parity_weights(cov: pd.DataFrame, upper_bounds: pd.Series | None = None) -> pd.Series:
    n = cov.shape[0]
    cov_np = cov.values

    if upper_bounds is None:
        upper_bounds = pd.Series(1.0, index=cov.index, dtype=float)

    def objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w, cov_np)
        target = rc.sum() / n
        return float(np.sum((rc - target) ** 2))

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, float(upper_bounds[a])) for a in cov.index]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Falló risk_parity_weights: {result.message}")

    weights = _normalize_long_only(result.x)
    return pd.Series(weights, index=cov.index, name="risk_parity")


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])

    while int(sort_ix.max()) >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clustered = sort_ix[sort_ix >= num_items]
        i = clustered.index
        j = clustered.values - num_items
        sort_ix.loc[i] = link[j, 0]
        sort_ix = pd.concat([sort_ix, pd.Series(link[j, 1], index=i + 1)])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _get_cluster_variance(cov: pd.DataFrame, cluster_items: List[str]) -> float:
    cov_slice = cov.loc[cluster_items, cluster_items]
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp = ivp / ivp.sum()
    return float(ivp @ cov_slice.values @ ivp)


def hrp_weights(returns_monthly: pd.DataFrame, upper_bounds: pd.Series | None = None) -> pd.Series:
    cov = returns_monthly.cov()
    corr = returns_monthly.corr()

    dist = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _get_quasi_diag(link)
    ordered_assets = returns_monthly.columns[sort_ix].tolist()

    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        left_cluster = cluster[:split]
        right_cluster = cluster[split:]

        left_var = _get_cluster_variance(cov, left_cluster)
        right_var = _get_cluster_variance(cov, right_cluster)

        alpha = 1.0 - left_var / (left_var + right_var)
        weights[left_cluster] *= alpha
        weights[right_cluster] *= (1.0 - alpha)

        clusters.append(left_cluster)
        clusters.append(right_cluster)

    weights = weights.reindex(returns_monthly.columns).fillna(0.0)
    weights = pd.Series(_normalize_long_only(weights.values), index=returns_monthly.columns, name="hrp")

    if upper_bounds is not None:
        weights = project_with_caps(weights, upper_bounds, name="hrp_caps")
        weights.name = "hrp"

    return weights


def build_base_models(
    asset_returns_monthly: pd.DataFrame,
    benchmark_returns_monthly: pd.Series,
    rf_monthly: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, float, float, pd.Series]:
    historical_mu_annual, historical_cov_annual = estimate_historical_mu_cov(asset_returns_monthly)
    betas_capm, mu_capm_annual, rf_annual, market_premium_annual = estimate_capm_expected_returns(
        asset_returns_monthly=asset_returns_monthly,
        benchmark_returns_monthly=benchmark_returns_monthly,
        rf_monthly=rf_monthly,
    )

    upper_bounds = get_upper_bounds(asset_returns_monthly.columns)

    models = {
        "mpt": maximum_sharpe_weights(
            historical_mu_annual,
            historical_cov_annual,
            rf=rf_annual,
            name="mpt",
            upper_bounds=upper_bounds,
        ),
        "capm_mvo": maximum_sharpe_weights(
            mu_capm_annual,
            historical_cov_annual,
            rf=rf_annual,
            name="capm_mvo",
            upper_bounds=upper_bounds,
        ),
        "risk_parity": risk_parity_weights(
            historical_cov_annual,
            upper_bounds=upper_bounds,
        ),
        "hrp": hrp_weights(
            asset_returns_monthly,
            upper_bounds=upper_bounds,
        ),
    }

    base_weights = pd.DataFrame(models).T
    base_weights.index.name = "base_model"
    return base_weights, historical_mu_annual, mu_capm_annual, historical_cov_annual, rf_annual, market_premium_annual, betas_capm


# =========================
# 4) ESCENARIOS ROBUSTOS ESTILO LÓPEZ DE PRADO
# =========================

def stationary_bootstrap_indices(
    n_obs: int,
    sample_length: int,
    avg_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs debe ser positivo.")
    if sample_length <= 0:
        raise ValueError("sample_length debe ser positivo.")
    avg_block_size = max(float(avg_block_size), 1.0)
    p = 1.0 / avg_block_size

    idx = np.empty(sample_length, dtype=int)
    idx[0] = int(rng.integers(0, n_obs))
    for t in range(1, sample_length):
        if float(rng.random()) < p:
            idx[t] = int(rng.integers(0, n_obs))
        else:
            idx[t] = (idx[t - 1] + 1) % n_obs
    return idx


def returns_to_price_paths(
    start_prices: pd.Series,
    returns_monthly: pd.DataFrame,
) -> pd.DataFrame:
    gross = (1.0 + returns_monthly).cumprod()
    return gross.mul(start_prices, axis=1)


def evaluate_static_weights(
    weights: pd.Series,
    asset_returns_monthly: pd.DataFrame,
    rf_monthly: pd.Series,
) -> pd.Series:
    weights = weights.reindex(asset_returns_monthly.columns).fillna(0.0)
    weights = pd.Series(_normalize_long_only(weights.values), index=asset_returns_monthly.columns)

    port_rets = asset_returns_monthly @ weights
    rf_aligned = rf_monthly.reindex(port_rets.index).ffill().fillna(0.0)
    excess = port_rets - rf_aligned

    n_months = int(port_rets.shape[0])
    total_return = float((1.0 + port_rets).prod() - 1.0)
    annual_return = float((1.0 + total_return) ** (12.0 / n_months) - 1.0) if n_months > 0 else np.nan
    annual_vol = float(port_rets.std(ddof=1) * np.sqrt(12.0)) if n_months > 1 else np.nan
    sharpe = float(excess.mean() * 12.0 / annual_vol) if annual_vol and annual_vol > 0 else np.nan

    wealth = (1.0 + port_rets).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    return pd.Series({
        "annual_return_oos": annual_return,
        "annual_volatility_oos": annual_vol,
        "sharpe_oos": sharpe,
        "max_drawdown_oos": max_drawdown,
        "total_return_oos": total_return,
    })


def robust_resampled_meta_inputs(
    asset_returns_monthly: pd.DataFrame,
    benchmark_returns_monthly: pd.Series,
    rf_monthly: pd.Series,
    asset_monthly_prices: pd.DataFrame,
    n_scenarios: int = 250,
    avg_block_size: float = 6.0,
    train_fraction: float = 0.7,
    random_state: int = 42,
    n_price_paths_to_store: int = 5,
) -> Dict[str, pd.DataFrame | pd.Series | float | list]:
    """
    Capa robusta inspirada en López de Prado:
    - genera escenarios sintéticos con stationary bootstrap sobre retornos mensuales conjuntos
    - reestima los modelos base sobre el tramo train de cada escenario
    - evalúa out-of-sample en el tramo test del mismo escenario
    - agrega:
        * pesos robustos por modelo (mediana entre escenarios)
        * mu / cov robustas (promedio entre escenarios)
        * alpha_center del meta-modelo según robustez out-of-sample
    """
    combined = pd.concat(
        [
            asset_returns_monthly,
            benchmark_returns_monthly.rename("__benchmark__"),
            rf_monthly.rename("__rf__"),
        ],
        axis=1,
    ).dropna(how="any")

    n_obs = int(combined.shape[0])
    if n_obs < 36:
        raise ValueError("Se requieren al menos 36 observaciones mensuales para la capa robusta.")

    train_size = int(round(n_obs * float(train_fraction)))
    train_size = max(train_size, 24)
    train_size = min(train_size, n_obs - 12)
    if train_size <= 0 or (n_obs - train_size) < 12:
        raise ValueError("No quedó un split train/test razonable para las simulaciones robustas.")

    rng = np.random.default_rng(random_state)
    asset_cols = list(asset_returns_monthly.columns)
    upper_bounds = get_upper_bounds(asset_cols)

    performance_records: List[dict] = []
    scenario_weights_records: List[dict] = []
    train_mu_list: List[pd.Series] = []
    train_cov_list: List[pd.DataFrame] = []
    train_mu_capm_list: List[pd.Series] = []
    train_betas_list: List[pd.Series] = []
    rf_annual_list: List[float] = []
    market_premium_list: List[float] = []
    failures: List[dict] = []
    scenario_price_samples: List[pd.DataFrame] = []

    start_prices = asset_monthly_prices.iloc[0].reindex(asset_cols).astype(float)

    for scenario_id in range(1, n_scenarios + 1):
        try:
            boot_idx = stationary_bootstrap_indices(
                n_obs=n_obs,
                sample_length=n_obs,
                avg_block_size=avg_block_size,
                rng=rng,
            )
            scenario = combined.iloc[boot_idx].reset_index(drop=True)
            scenario.index = combined.index  # conserva una grilla mensual regular

            scenario_assets = scenario[asset_cols].copy()
            scenario_benchmark = scenario["__benchmark__"].copy()
            scenario_rf = scenario["__rf__"].copy()

            train_assets = scenario_assets.iloc[:train_size].copy()
            train_benchmark = scenario_benchmark.iloc[:train_size].copy()
            train_rf = scenario_rf.iloc[:train_size].copy()

            test_assets = scenario_assets.iloc[train_size:].copy()
            test_rf = scenario_rf.iloc[train_size:].copy()

            if train_assets.shape[0] < 24 or test_assets.shape[0] < 12:
                raise ValueError("Escenario con split train/test insuficiente.")

            base_weights_s, hist_mu_s, mu_capm_s, cov_s, rf_annual_s, market_premium_s, betas_s = build_base_models(
                asset_returns_monthly=train_assets,
                benchmark_returns_monthly=train_benchmark,
                rf_monthly=train_rf,
            )

            train_mu_list.append(hist_mu_s)
            train_cov_list.append(cov_s)
            train_mu_capm_list.append(mu_capm_s)
            train_betas_list.append(betas_s)
            rf_annual_list.append(float(rf_annual_s))
            market_premium_list.append(float(market_premium_s))

            for model_name, row in base_weights_s.iterrows():
                weights = row.reindex(asset_cols).astype(float)
                eval_stats = evaluate_static_weights(weights, test_assets, test_rf)
                record = {
                    "scenario_id": scenario_id,
                    "base_model": model_name,
                }
                record.update(eval_stats.to_dict())
                performance_records.append(record)

                weight_row = {"scenario_id": scenario_id, "base_model": model_name}
                for asset in asset_cols:
                    weight_row[f"w_{asset}"] = float(weights[asset])
                scenario_weights_records.append(weight_row)

            if scenario_id <= n_price_paths_to_store:
                scenario_prices = returns_to_price_paths(start_prices, scenario_assets)
                scenario_prices = scenario_prices.reset_index().rename(columns={"index": "date"})
                scenario_prices.insert(0, "scenario_id", scenario_id)
                scenario_price_samples.append(scenario_prices)

        except Exception as exc:
            failures.append({"scenario_id": scenario_id, "error": str(exc)})

    if not performance_records:
        raise RuntimeError("No hubo escenarios robustos válidos. Revisá los datos o bajá la complejidad.")

    scenario_oos = pd.DataFrame(performance_records)
    scenario_weights = pd.DataFrame(scenario_weights_records)

    # Pesos robustos de cada modelo: mediana entre escenarios
    robust_base_weights = (
        scenario_weights.groupby("base_model")[[c for c in scenario_weights.columns if c.startswith("w_")]]
        .median()
    )
    robust_base_weights.columns = [c.replace("w_", "", 1) for c in robust_base_weights.columns]
    robust_base_weights = robust_base_weights.reindex(columns=asset_cols)

    for model_name in robust_base_weights.index:
        row = robust_base_weights.loc[model_name].copy()
        row = pd.Series(np.maximum(row.values, 0.0), index=row.index, name=model_name)
        row = project_with_caps(row, upper_bounds, name=f"{model_name}_robust_caps")
        robust_base_weights.loc[model_name] = row.values

    # Inputs robustos para la evaluación final del meta-modelo
    robust_mu_annual = pd.concat(train_mu_list, axis=1).T.median(axis=0)
    robust_mu_annual.name = "robust_mu_annual"

    robust_mu_capm_annual = pd.concat(train_mu_capm_list, axis=1).T.median(axis=0)
    robust_mu_capm_annual.name = "robust_mu_capm_annual"

    robust_betas_capm = pd.concat(train_betas_list, axis=1).T.median(axis=0)
    robust_betas_capm.name = "robust_beta_capm"

    robust_cov_annual = sum(train_cov_list) / float(len(train_cov_list))
    robust_cov_annual = robust_cov_annual.reindex(index=asset_cols, columns=asset_cols)

    robust_rf_annual = float(np.mean(rf_annual_list))
    robust_market_premium_annual = float(np.mean(market_premium_list))

    # Dispersión de pesos por modelo
    dispersion_records = []
    for model_name in scenario_weights["base_model"].unique():
        sub = scenario_weights.loc[scenario_weights["base_model"] == model_name].copy()
        w_cols = [c for c in sub.columns if c.startswith("w_")]
        median_row = sub[w_cols].median(axis=0)
        l1_disp = 0.5 * sub[w_cols].sub(median_row, axis=1).abs().sum(axis=1)
        dispersion_records.append({
            "base_model": model_name,
            "mean_weight_l1_dispersion": float(l1_disp.mean()),
            "median_weight_l1_dispersion": float(l1_disp.median()),
        })
    dispersion_df = pd.DataFrame(dispersion_records).set_index("base_model")

    robust_model_summary = scenario_oos.groupby("base_model").agg(
        mean_sharpe_oos=("sharpe_oos", "mean"),
        median_sharpe_oos=("sharpe_oos", "median"),
        std_sharpe_oos=("sharpe_oos", "std"),
        prob_positive_sharpe=("sharpe_oos", lambda x: float(np.mean(np.asarray(x) > 0))),
        mean_annual_return_oos=("annual_return_oos", "mean"),
        mean_annual_volatility_oos=("annual_volatility_oos", "mean"),
        mean_max_drawdown_oos=("max_drawdown_oos", "mean"),
        n_valid_scenarios=("sharpe_oos", "count"),
    )
    robust_model_summary = robust_model_summary.join(dispersion_df, how="left")

    robust_model_summary["robust_score_raw"] = (
        np.maximum(robust_model_summary["mean_sharpe_oos"], 0.0)
        * np.maximum(robust_model_summary["prob_positive_sharpe"], 0.0)
        / (1.0 + robust_model_summary["mean_weight_l1_dispersion"].fillna(0.0))
    )

    raw_sum = float(robust_model_summary["robust_score_raw"].sum())
    if raw_sum <= 0:
        robust_model_summary["alpha_center"] = 1.0 / len(robust_model_summary)
    else:
        robust_model_summary["alpha_center"] = robust_model_summary["robust_score_raw"] / raw_sum

    robust_model_summary = robust_model_summary.reset_index()

    scenario_price_paths_sample = (
        pd.concat(scenario_price_samples, axis=0, ignore_index=True)
        if scenario_price_samples
        else pd.DataFrame()
    )

    return {
        "robust_base_weights": robust_base_weights,
        "robust_mu_annual": robust_mu_annual,
        "robust_mu_capm_annual": robust_mu_capm_annual,
        "robust_cov_annual": robust_cov_annual,
        "robust_betas_capm": robust_betas_capm,
        "robust_rf_annual": robust_rf_annual,
        "robust_market_premium_annual": robust_market_premium_annual,
        "robust_model_summary": robust_model_summary,
        "scenario_oos_results": scenario_oos,
        "scenario_weights_long": scenario_weights,
        "scenario_price_paths_sample": scenario_price_paths_sample,
        "scenario_failures": failures,
        "scenario_train_size": train_size,
        "scenario_n_obs": n_obs,
    }


# =========================
# 5) META-MODELO MONTE CARLO
# =========================

def simulate_meta_model(
    base_model_weights: pd.DataFrame,
    evaluation_mu_annual: pd.Series,
    evaluation_cov_annual: pd.DataFrame,
    n_simulations: int = 50_000,
    rf_annual: float = 0.0,
    random_state: int = 42,
    alpha_center: pd.Series | None = None,
    dirichlet_strength: float = 25.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_models = base_model_weights.shape[0]

    if alpha_center is None:
        concentration = np.ones(n_models)
    else:
        alpha_center = alpha_center.reindex(base_model_weights.index).fillna(0.0)
        if float(alpha_center.sum()) <= 0:
            alpha_center = pd.Series(1.0 / n_models, index=base_model_weights.index)
        else:
            alpha_center = alpha_center / float(alpha_center.sum())
        concentration = np.maximum(alpha_center.values * float(dirichlet_strength), 1e-3)

    alpha_samples = rng.dirichlet(concentration, size=n_simulations)
    final_weights = alpha_samples @ base_model_weights.values

    mu_np = evaluation_mu_annual.reindex(base_model_weights.columns).values
    cov_np = evaluation_cov_annual.reindex(index=base_model_weights.columns, columns=base_model_weights.columns).values

    expected_returns = final_weights @ mu_np
    volatilities = np.sqrt(np.einsum("ij,jk,ik->i", final_weights, cov_np, final_weights))
    sharpe = np.divide(
        expected_returns - rf_annual,
        volatilities,
        out=np.full_like(expected_returns, np.nan, dtype=float),
        where=volatilities > 0,
    )

    results = pd.DataFrame({
        "expected_return": expected_returns,
        "volatility": volatilities,
        "sharpe": sharpe,
    })

    for i, model_name in enumerate(base_model_weights.index):
        results[f"alpha_{model_name}"] = alpha_samples[:, i]

    for i, asset_name in enumerate(base_model_weights.columns):
        results[f"w_{asset_name}"] = final_weights[:, i]

    return results


# =========================
# 6) FRONTERA EFICIENTE + BUCKETS
# =========================

def efficient_frontier(portfolios: pd.DataFrame) -> pd.DataFrame:
    ordered = portfolios.sort_values(["volatility", "expected_return"], ascending=[True, False]).copy()

    keep_idx: List[int] = []
    best_return_so_far = -np.inf
    for idx, row in ordered.iterrows():
        if row["expected_return"] > best_return_so_far + 1e-12:
            keep_idx.append(idx)
            best_return_so_far = row["expected_return"]

    frontier = ordered.loc[keep_idx].sort_values("volatility").reset_index(drop=True)
    return frontier


def sample_frontier(frontier: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    if frontier.empty:
        raise ValueError("La frontera eficiente quedó vacía.")

    if len(frontier) <= n_points:
        sampled = frontier.copy().reset_index(drop=True)
    else:
        idx = np.linspace(0, len(frontier) - 1, num=n_points, dtype=int)
        idx = np.unique(idx)
        sampled = frontier.iloc[idx].reset_index(drop=True)

    sampled["frontier_rank"] = np.arange(1, len(sampled) + 1)
    sampled["frontier_pct"] = sampled["frontier_rank"] / len(sampled)
    return sampled


def select_profile_portfolios(frontier_sampled: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "muy_conservador",
        "conservador",
        "moderado",
        "levemente_riesgoso",
        "muy_riesgoso",
    ]

    bucket_indices = np.array_split(np.arange(len(frontier_sampled)), len(labels))
    selected_rows = []

    for label, idx_group in zip(labels, bucket_indices):
        bucket = frontier_sampled.iloc[idx_group].copy()
        if bucket.empty:
            continue
        best_row = bucket.loc[bucket["sharpe"].idxmax()].copy()
        best_row["profile_bucket"] = label
        best_row["bucket_size"] = len(bucket)
        selected_rows.append(best_row)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


# =========================
# 7) BACKTEST CON REBALANCEO MENSUAL Y BANDAS
# =========================

def _extract_target_weights_from_row(row: pd.Series, asset_columns: List[str]) -> pd.Series:
    weights = pd.Series({asset: float(row[f"w_{asset}"]) for asset in asset_columns}, dtype=float)
    weights = pd.Series(_normalize_long_only(weights.values), index=weights.index)
    return weights


def simulate_band_rebalanced_portfolio(
    asset_returns_monthly: pd.DataFrame,
    target_weights: pd.Series,
    band: float = 0.03,
    initial_value: float = 100.0,
) -> pd.DataFrame:
    asset_columns = list(asset_returns_monthly.columns)
    target = target_weights.reindex(asset_columns).fillna(0.0)
    target = pd.Series(_normalize_long_only(target.values), index=asset_columns)

    current_weights = target.copy()
    portfolio_value = float(initial_value)
    records: List[dict] = []

    for date, month_ret in asset_returns_monthly.iterrows():
        month_ret = month_ret.reindex(asset_columns).astype(float)
        start_weights = current_weights.copy()
        portfolio_return = float(start_weights @ month_ret)
        portfolio_value *= (1.0 + portfolio_return)

        gross_asset_values = start_weights * (1.0 + month_ret)
        gross_total = float(gross_asset_values.sum())
        if gross_total <= 0:
            raise ValueError(f"La cartera colapsó o quedó con valor no positivo en {date}.")

        post_return_weights = gross_asset_values / gross_total
        drift = post_return_weights - target

        cap_breach = (
            ("BTC" in post_return_weights.index and float(post_return_weights["BTC"]) > 0.05 + 1e-12)
            or ("GLD" in post_return_weights.index and float(post_return_weights["GLD"]) > 0.15 + 1e-12)
        )
        rebalance_flag = bool((drift.abs() > band + 1e-12).any() or cap_breach)
        turnover = float(0.5 * np.abs(target.values - post_return_weights.values).sum()) if rebalance_flag else 0.0
        end_weights = target.copy() if rebalance_flag else post_return_weights.copy()

        row = {
            "date": date,
            "portfolio_return": portfolio_return,
            "portfolio_value": portfolio_value,
            "cumulative_return": portfolio_value / initial_value - 1.0,
            "rebalance_flag": int(rebalance_flag),
            "turnover": turnover,
        }

        for asset in asset_columns:
            row[f"target_w_{asset}"] = float(target[asset])
            row[f"start_w_{asset}"] = float(start_weights[asset])
            row[f"post_w_{asset}"] = float(post_return_weights[asset])
            row[f"end_w_{asset}"] = float(end_weights[asset])
            row[f"drift_{asset}"] = float(drift[asset])

        records.append(row)
        current_weights = end_weights

    backtest = pd.DataFrame(records)
    if not backtest.empty:
        backtest["date"] = pd.to_datetime(backtest["date"])
        backtest = backtest.sort_values("date").reset_index(drop=True)
    return backtest


def summarize_backtest(
    backtest_df: pd.DataFrame,
    rf_monthly: pd.Series,
    profile_name: str,
) -> pd.Series:
    if backtest_df.empty:
        raise ValueError(f"El backtest del perfil '{profile_name}' quedó vacío.")

    monthly_returns = backtest_df.set_index("date")["portfolio_return"].copy()
    rf_aligned = rf_monthly.reindex(monthly_returns.index).astype(float)
    excess_returns = monthly_returns - rf_aligned

    n_months = int(monthly_returns.shape[0])
    final_value = float(backtest_df["portfolio_value"].iloc[-1])
    total_return = final_value / float(backtest_df["portfolio_value"].iloc[0] / (1.0 + backtest_df["portfolio_return"].iloc[0])) - 1.0
    annual_return = float((1.0 + total_return) ** (12.0 / n_months) - 1.0) if n_months > 0 else np.nan
    annual_vol = float(monthly_returns.std(ddof=1) * np.sqrt(12.0)) if n_months > 1 else np.nan
    annual_excess = float(excess_returns.mean() * 12.0)
    sharpe = annual_excess / annual_vol if annual_vol and annual_vol > 0 else np.nan

    wealth_index = (1.0 + monthly_returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1.0
    max_drawdown = float(drawdown.min())

    avg_turnover_when_rebalanced = float(backtest_df.loc[backtest_df["rebalance_flag"] == 1, "turnover"].mean())
    annual_turnover_average = float(backtest_df["turnover"].sum() * (12.0 / n_months)) if n_months > 0 else np.nan

    return pd.Series({
        "profile_bucket": profile_name,
        "months": n_months,
        "annual_return_backtest": annual_return,
        "annual_volatility_backtest": annual_vol,
        "sharpe_backtest": sharpe,
        "total_return_backtest": total_return,
        "max_drawdown_backtest": max_drawdown,
        "final_value_backtest": final_value,
        "n_rebalances": int(backtest_df["rebalance_flag"].sum()),
        "avg_turnover_when_rebalanced": avg_turnover_when_rebalanced,
        "annual_turnover_average": annual_turnover_average,
    })


def backtest_selected_profiles(
    selected_profiles: pd.DataFrame,
    asset_returns_monthly: pd.DataFrame,
    rf_monthly: pd.Series,
    band: float = 0.03,
    initial_value: float = 100.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    asset_columns = list(asset_returns_monthly.columns)
    profile_paths: List[pd.DataFrame] = []
    profile_summaries: List[pd.Series] = []

    for _, row in selected_profiles.iterrows():
        profile_name = str(row["profile_bucket"])
        target_weights = _extract_target_weights_from_row(row, asset_columns)
        backtest = simulate_band_rebalanced_portfolio(
            asset_returns_monthly=asset_returns_monthly,
            target_weights=target_weights,
            band=band,
            initial_value=initial_value,
        )
        backtest.insert(1, "profile_bucket", profile_name)
        profile_paths.append(backtest)
        profile_summaries.append(summarize_backtest(backtest, rf_monthly=rf_monthly, profile_name=profile_name))

    backtest_long = pd.concat(profile_paths, axis=0, ignore_index=True) if profile_paths else pd.DataFrame()
    backtest_summary = pd.DataFrame(profile_summaries).reset_index(drop=True) if profile_summaries else pd.DataFrame()
    return backtest_long, backtest_summary


# =========================
# 8) VALIDACIONES / ALERTAS
# =========================

def generate_warnings(
    prices_daily: pd.DataFrame,
    asset_cols: List[str],
    benchmark_col: str,
    rf_col: str,
    benchmark_returns_monthly: pd.Series,
    asset_returns_monthly: pd.DataFrame,
    scenario_failures: List[dict] | None = None,
) -> List[str]:
    warnings: List[str] = []

    if benchmark_col in asset_cols:
        warnings.append(f"La columna de benchmark '{benchmark_col}' también está dentro del universo invertible.")
    if rf_col in asset_cols:
        warnings.append(f"La columna de RF '{rf_col}' también está dentro del universo invertible.")

    for asset in asset_cols:
        if asset in prices_daily.columns:
            left = prices_daily[asset].dropna()
            right = prices_daily[benchmark_col].dropna()
            common = left.index.intersection(right.index)
            if len(common) > 0:
                diff = (left.loc[common] - right.loc[common]).abs()
                if float(diff.max()) == 0.0:
                    warnings.append(
                        f"El benchmark '{benchmark_col}' es idéntico a la serie del activo '{asset}' en todo el tramo común."
                    )
                    break

    if benchmark_returns_monthly.var(ddof=1) <= 0:
        warnings.append("La varianza del benchmark mensual es no positiva; revisar la serie de benchmark.")

    if asset_returns_monthly.isna().any().any():
        warnings.append("Persisten NaN en retornos mensuales de activos luego de la limpieza.")

    if scenario_failures:
        warnings.append(f"Hubo {len(scenario_failures)} escenarios robustos descartados por errores numéricos o datos.")

    return warnings


# =========================
# 9) PIPELINE COMPLETO
# =========================

def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    sheet_name: str | int = 0,
    date_col: str = "FECHA",
    rf_col: str = "T-BILL10",
    benchmark_col: str = "60/40",
    n_simulations: int = 50_000,
    frontier_points: int = 100,
    random_state: int = 42,
    rebalance_band: float = 0.03,
    initial_value: float = 100.0,
    n_scenarios: int = 250,
    avg_block_size: float = 6.0,
    train_fraction: float = 0.7,
    dirichlet_strength: float = 25.0,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_bloomberg_dataset(input_path, sheet_name=sheet_name, date_col=date_col)

    if rf_col not in df.columns:
        raise ValueError(f"No encontré la columna de RF '{rf_col}'.")
    if benchmark_col not in df.columns:
        raise ValueError(f"No encontré la columna de benchmark '{benchmark_col}'.")

    asset_cols = [c for c in df.columns if c not in {rf_col, benchmark_col}]
    if not asset_cols:
        raise ValueError("No quedaron activos invertibles luego de excluir benchmark y RF.")

    prices_daily = prepare_price_matrix(df, asset_cols)
    benchmark_daily = prepare_price_matrix(df, [benchmark_col])[benchmark_col]
    rf_daily = df[rf_col].copy()

    monthly_prices, asset_returns_monthly_raw = daily_prices_to_monthly_returns(prices_daily)
    benchmark_monthly_prices, benchmark_returns_monthly_raw = daily_prices_to_monthly_returns(benchmark_daily.to_frame())
    benchmark_returns_monthly_raw = benchmark_returns_monthly_raw[benchmark_col]
    monthly_rf_yield_pct, rf_monthly_raw = daily_yield_to_monthly_rf_returns(rf_daily)

    asset_returns_monthly, benchmark_returns_monthly, rf_monthly = align_monthly_inputs(
        asset_returns=asset_returns_monthly_raw,
        benchmark_returns=benchmark_returns_monthly_raw,
        rf_monthly=rf_monthly_raw,
    )

    # Modelos base históricos (solo para comparación)
    hist_base_model_weights, hist_mu_annual, hist_mu_capm_annual, hist_cov_annual, hist_rf_annual, hist_market_premium_annual, hist_betas_capm = build_base_models(
        asset_returns_monthly=asset_returns_monthly,
        benchmark_returns_monthly=benchmark_returns_monthly,
        rf_monthly=rf_monthly,
    )

    # Capa robusta inspirada en López de Prado
    robust_outputs = robust_resampled_meta_inputs(
        asset_returns_monthly=asset_returns_monthly,
        benchmark_returns_monthly=benchmark_returns_monthly,
        rf_monthly=rf_monthly,
        asset_monthly_prices=monthly_prices,
        n_scenarios=n_scenarios,
        avg_block_size=avg_block_size,
        train_fraction=train_fraction,
        random_state=random_state,
        n_price_paths_to_store=min(5, max(1, n_scenarios)),
    )

    base_model_weights = robust_outputs["robust_base_weights"]
    historical_mu_annual = robust_outputs["robust_mu_annual"]
    mu_capm_annual = robust_outputs["robust_mu_capm_annual"]
    historical_cov_annual = robust_outputs["robust_cov_annual"]
    rf_annual = float(robust_outputs["robust_rf_annual"])
    market_premium_annual = float(robust_outputs["robust_market_premium_annual"])
    betas_capm = robust_outputs["robust_betas_capm"]

    alpha_center = (
        robust_outputs["robust_model_summary"]
        .set_index("base_model")["alpha_center"]
        .reindex(base_model_weights.index)
    )

    warnings = generate_warnings(
        prices_daily=df,
        asset_cols=asset_cols,
        benchmark_col=benchmark_col,
        rf_col=rf_col,
        benchmark_returns_monthly=benchmark_returns_monthly,
        asset_returns_monthly=asset_returns_monthly,
        scenario_failures=robust_outputs["scenario_failures"],
    )

    all_meta_portfolios = simulate_meta_model(
        base_model_weights=base_model_weights,
        evaluation_mu_annual=historical_mu_annual,
        evaluation_cov_annual=historical_cov_annual,
        n_simulations=n_simulations,
        rf_annual=rf_annual,
        random_state=random_state,
        alpha_center=alpha_center,
        dirichlet_strength=dirichlet_strength,
    )

    frontier = efficient_frontier(all_meta_portfolios)
    frontier_sampled = sample_frontier(frontier, n_points=frontier_points)
    selected_profiles = select_profile_portfolios(frontier_sampled)
    profile_backtest_long, profile_backtest_summary = backtest_selected_profiles(
        selected_profiles=selected_profiles,
        asset_returns_monthly=asset_returns_monthly,
        rf_monthly=rf_monthly,
        band=rebalance_band,
        initial_value=initial_value,
    )

    prices_daily.to_csv(output_dir / "01_clean_daily_prices_assets.csv", index=True)
    monthly_prices.to_csv(output_dir / "02_monthly_prices_assets.csv", index=True)
    asset_returns_monthly.to_csv(output_dir / "03_monthly_returns_assets.csv", index=True)
    benchmark_monthly_prices.to_csv(output_dir / "04_monthly_prices_benchmark.csv", index=True)
    benchmark_returns_monthly.to_frame(name="benchmark_return").to_csv(output_dir / "05_monthly_returns_benchmark.csv", index=True)
    monthly_rf_yield_pct.to_frame(name="rf_yield_pct_eom").to_csv(output_dir / "06_monthly_rf_yield_pct.csv", index=True)
    rf_monthly.to_frame(name="rf_monthly_return").to_csv(output_dir / "07_monthly_rf_return_aligned.csv", index=True)

    hist_mu_annual.to_frame(name="expected_return_historical_annual_single_path").to_csv(output_dir / "08_mu_historical_annual_single_path.csv", index=True)
    hist_cov_annual.to_csv(output_dir / "09_cov_historical_annual_single_path.csv", index=True)
    hist_betas_capm.to_frame(name="beta_capm_single_path").to_csv(output_dir / "10_capm_betas_single_path.csv", index=True)
    hist_mu_capm_annual.to_frame(name="expected_return_capm_annual_single_path").to_csv(output_dir / "11_mu_capm_annual_single_path.csv", index=True)
    hist_base_model_weights.to_csv(output_dir / "12_base_model_weights_single_path.csv", index=True)

    historical_mu_annual.to_frame(name="expected_return_historical_annual_robust").to_csv(output_dir / "13_mu_historical_annual_robust.csv", index=True)
    historical_cov_annual.to_csv(output_dir / "14_cov_historical_annual_robust.csv", index=True)
    betas_capm.to_frame(name="beta_capm_robust").to_csv(output_dir / "15_capm_betas_robust.csv", index=True)
    mu_capm_annual.to_frame(name="expected_return_capm_annual_robust").to_csv(output_dir / "16_mu_capm_annual_robust.csv", index=True)
    base_model_weights.to_csv(output_dir / "17_base_model_weights_robust.csv", index=True)
    robust_outputs["robust_model_summary"].to_csv(output_dir / "18_lp_model_summary.csv", index=False)
    robust_outputs["scenario_oos_results"].to_csv(output_dir / "19_lp_scenario_model_oos.csv", index=False)
    robust_outputs["scenario_weights_long"].to_csv(output_dir / "20_lp_scenario_weights_long.csv", index=False)
    robust_outputs["scenario_price_paths_sample"].to_csv(output_dir / "21_lp_scenario_price_paths_sample.csv", index=False)

    all_meta_portfolios.to_csv(output_dir / "22_all_meta_portfolios.csv", index=False)
    frontier.to_csv(output_dir / "23_efficient_frontier_full.csv", index=False)
    frontier_sampled.to_csv(output_dir / "24_efficient_frontier_sampled.csv", index=False)
    selected_profiles.to_csv(output_dir / "25_selected_profiles.csv", index=False)
    profile_backtest_long.to_csv(output_dir / "26_profile_backtest_monthly.csv", index=False)
    profile_backtest_summary.to_csv(output_dir / "27_profile_backtest_summary.csv", index=False)

    summary = {
        "input_file": str(input_path),
        "sheet_name": sheet_name,
        "date_col": date_col,
        "rf_col": rf_col,
        "benchmark_col": benchmark_col,
        "n_assets": int(len(asset_cols)),
        "assets": asset_cols,
        "first_date_assets": str(prices_daily.index.min().date()),
        "last_date_assets": str(prices_daily.index.max().date()),
        "n_daily_obs_assets": int(prices_daily.shape[0]),
        "n_monthly_obs": int(asset_returns_monthly.shape[0]),
        "n_base_models": int(base_model_weights.shape[0]),
        "base_models": base_model_weights.index.tolist(),
        "n_scenarios_lp": int(n_scenarios),
        "avg_block_size_lp": float(avg_block_size),
        "train_fraction_lp": float(train_fraction),
        "scenario_train_size_lp": int(robust_outputs["scenario_train_size"]),
        "scenario_failures_lp": robust_outputs["scenario_failures"],
        "n_simulations": int(n_simulations),
        "dirichlet_strength": float(dirichlet_strength),
        "frontier_size_full": int(len(frontier)),
        "frontier_size_sampled": int(len(frontier_sampled)),
        "profiles_selected": selected_profiles["profile_bucket"].tolist(),
        "rf_annual_used_robust": rf_annual,
        "market_premium_annual_used_robust": market_premium_annual,
        "rebalance_frequency": "monthly_review",
        "drift_band_abs": rebalance_band,
        "backtest_initial_value": initial_value,
        "warnings": warnings,
    }
    with open(output_dir / "00_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "prices_daily": prices_daily,
        "monthly_prices": monthly_prices,
        "asset_returns_monthly": asset_returns_monthly,
        "benchmark_returns_monthly": benchmark_returns_monthly.to_frame(name=benchmark_col),
        "rf_monthly": rf_monthly.to_frame(name=rf_col),
        "historical_mu_annual": historical_mu_annual.to_frame(name="historical_mu_annual_robust"),
        "historical_cov_annual": historical_cov_annual,
        "mu_capm_annual": mu_capm_annual.to_frame(name="capm_mu_annual_robust"),
        "betas_capm": betas_capm.to_frame(name="beta_capm_robust"),
        "base_model_weights": base_model_weights,
        "robust_model_summary": robust_outputs["robust_model_summary"],
        "scenario_oos_results": robust_outputs["scenario_oos_results"],
        "all_meta_portfolios": all_meta_portfolios,
        "frontier": frontier,
        "frontier_sampled": frontier_sampled,
        "selected_profiles": selected_profiles,
        "profile_backtest_long": profile_backtest_long,
        "profile_backtest_summary": profile_backtest_summary,
    }


# =========================
# 10) CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-modelo de strategic allocation con capa robusta estilo López de Prado: escenarios sintéticos, MPT, CAPM-MVO, Risk Parity y HRP, más rebalanceo mensual con bandas."
    )
    parser.add_argument("--input", required=True, help="Ruta al CSV o Excel de Bloomberg")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument("--sheet-name", default=0, help="Nombre o índice de hoja. Solo aplica a Excel.")
    parser.add_argument("--date-col", default="FECHA", help="Nombre de la columna de fecha")
    parser.add_argument("--rf-col", default="T-BILL10", help="Columna de tasa libre de riesgo")
    parser.add_argument("--benchmark-col", default="60/40", help="Columna de benchmark CAPM")
    parser.add_argument("--n-simulations", type=int, default=50_000, help="Cantidad de simulaciones Monte Carlo del meta-modelo")
    parser.add_argument("--frontier-points", type=int, default=100, help="Cantidad de puntos a conservar en la frontera muestreada")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--rebalance-band", type=float, default=0.03, help="Banda absoluta de drift por activo. Ej.: 0.03 = ±3 p.p.")
    parser.add_argument("--initial-value", type=float, default=100.0, help="Valor inicial para el backtest")
    parser.add_argument("--n-scenarios", type=int, default=250, help="Cantidad de escenarios sintéticos robustos")
    parser.add_argument("--avg-block-size", type=float, default=6.0, help="Tamaño promedio de bloque del stationary bootstrap (en meses)")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Proporción train de cada escenario")
    parser.add_argument("--dirichlet-strength", type=float, default=25.0, help="Fuerza del prior robusto sobre los pesos del meta-modelo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        sheet_name = int(args.sheet_name)
    except (TypeError, ValueError):
        sheet_name = args.sheet_name

    outputs = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        sheet_name=sheet_name,
        date_col=args.date_col,
        rf_col=args.rf_col,
        benchmark_col=args.benchmark_col,
        n_simulations=args.n_simulations,
        frontier_points=args.frontier_points,
        random_state=args.random_state,
        rebalance_band=args.rebalance_band,
        initial_value=args.initial_value,
        n_scenarios=args.n_scenarios,
        avg_block_size=args.avg_block_size,
        train_fraction=args.train_fraction,
        dirichlet_strength=args.dirichlet_strength,
    )

    selected = outputs["selected_profiles"][[
        "profile_bucket",
        "expected_return",
        "volatility",
        "sharpe",
    ]]
    print("\nPerfiles seleccionados (frontera robusta):\n")
    print(selected.to_string(index=False))

    robust_summary = outputs["robust_model_summary"][[
        "base_model",
        "mean_sharpe_oos",
        "prob_positive_sharpe",
        "mean_weight_l1_dispersion",
        "alpha_center",
    ]]
    print("\nResumen de robustez estilo López de Prado:\n")
    print(robust_summary.to_string(index=False))

    if not outputs["profile_backtest_summary"].empty:
        print("\nResumen backtest con rebalanceo mensual y bandas:\n")
        cols = [
            "profile_bucket",
            "annual_return_backtest",
            "annual_volatility_backtest",
            "sharpe_backtest",
            "max_drawdown_backtest",
            "n_rebalances",
        ]
        print(outputs["profile_backtest_summary"][cols].to_string(index=False))

    print(f"\nArchivos exportados en: {Path(args.output_dir).resolve()}")


def run_notebook(
    input_path: str | Path,
    output_dir: str | Path,
    sheet_name: str | int = 0,
    date_col: str = "FECHA",
    rf_col: str = "T-BILL10",
    benchmark_col: str = "60/40",
    n_simulations: int = 50_000,
    frontier_points: int = 100,
    random_state: int = 42,
    rebalance_band: float = 0.03,
    initial_value: float = 100.0,
    n_scenarios: int = 250,
    avg_block_size: float = 6.0,
    train_fraction: float = 0.7,
    dirichlet_strength: float = 25.0,
) -> Dict[str, pd.DataFrame]:
    return run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        sheet_name=sheet_name,
        date_col=date_col,
        rf_col=rf_col,
        benchmark_col=benchmark_col,
        n_simulations=n_simulations,
        frontier_points=frontier_points,
        random_state=random_state,
        rebalance_band=rebalance_band,
        initial_value=initial_value,
        n_scenarios=n_scenarios,
        avg_block_size=avg_block_size,
        train_fraction=train_fraction,
        dirichlet_strength=dirichlet_strength,
    )


if __name__ == "__main__":
    main()
