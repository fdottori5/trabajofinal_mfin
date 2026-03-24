from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.pop("MPLBACKEND", None)
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de estilo académico global
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.frameon": True,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Paleta de colores profesional
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

PROFILE_ORDER = [
    "muy_conservador",
    "conservador",
    "moderado",
    "levemente_riesgoso",
    "muy_riesgoso",
]
BASE_MODEL_ORDER = ["mpt", "capm_mvo", "risk_parity", "hrp"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _resolve_path(output_dir: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = output_dir / name
        if p.exists():
            return p
    return None

def _load_csv(output_dir: Path, candidates: list[str]) -> Optional[pd.DataFrame]:
    p = _resolve_path(output_dir, candidates)
    if p is None:
        return None
    return pd.read_csv(p)

def _load_bundle(output_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "base_single": _load_csv(output_dir, ["12_base_model_weights_single_path.csv", "12_base_model_weights.csv"]),
        "base_robust": _load_csv(output_dir, ["17_base_model_weights_robust.csv", "17_base_model_weights.csv", "12b_base_model_weights_resampled_mean.csv", "12_base_model_weights.csv"]),
        "lp_summary": _load_csv(output_dir, ["18_lp_model_summary.csv"]),
        "scenario_oos": _load_csv(output_dir, ["19_lp_scenario_model_oos.csv"]),
        "scenario_weights": _load_csv(output_dir, ["20_lp_scenario_weights_long.csv"]),
        "scenario_paths": _load_csv(output_dir, ["21_lp_scenario_price_paths_sample.csv"]),
        "all_meta": _load_csv(output_dir, ["22_all_meta_portfolios.csv", "13_all_meta_portfolios.csv"]),
        "frontier_full": _load_csv(output_dir, ["23_efficient_frontier_full.csv", "14_efficient_frontier_full.csv"]),
        "frontier_sampled": _load_csv(output_dir, ["24_efficient_frontier_sampled.csv", "15_efficient_frontier_sampled.csv"]),
        "selected": _load_csv(output_dir, ["25_selected_profiles.csv", "16_selected_profiles.csv"]),
        "monthly": _load_csv(output_dir, ["26_profile_backtest_monthly.csv", "17_profile_backtest_monthly.csv"]),
        "summary": _load_csv(output_dir, ["27_profile_backtest_summary.csv", "18_profile_backtest_summary.csv"]),
        "betas_single": _load_csv(output_dir, ["10_capm_betas_single_path.csv", "10_capm_betas.csv"]),
        "betas_robust": _load_csv(output_dir, ["15_capm_betas_robust.csv"]),
    }

def _ordered_profiles(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str)
    present = [p for p in PROFILE_ORDER if p in set(vals)]
    others = [p for p in vals.unique().tolist() if p not in present]
    return present + others

def _asset_weight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("w_")]

def _alpha_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("alpha_")]

def _set_profile_order(df: pd.DataFrame, col: str = "profile_bucket") -> pd.DataFrame:
    ordered = _ordered_profiles(df[col])
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=ordered, ordered=True)
    return out.sort_values(col)

def _clean_beta_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    first = out.columns[0]
    if first != "asset":
        out = out.rename(columns={first: "asset"})
    beta_cols = [c for c in out.columns if c != "asset"]
    if len(beta_cols) == 1 and beta_cols[0] != "beta_capm":
        out = out.rename(columns={beta_cols[0]: "beta_capm"})
    return out[["asset", "beta_capm"]]

# --- Funciones de Gráficos (Mejorado el formato) ---

def plot_base_model_weights(df: pd.DataFrame, outpath: Path, title: str) -> None:
    plot_df = df.copy().set_index("base_model")
    plot_df = plot_df.reindex([m for m in BASE_MODEL_ORDER if m in plot_df.index])
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(10, 6), color=COLORS)
    ax.set_title(title, pad=20)
    ax.set_xlabel("Modelo base")
    ax.set_ylabel("Asignación de Peso")
    ax.legend(title="Activo", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_base_model_weights_comparison(single_df: pd.DataFrame, robust_df: pd.DataFrame, outpath: Path) -> None:
    left = single_df.copy().set_index("base_model")
    right = robust_df.copy().set_index("base_model")
    common_models = [m for m in BASE_MODEL_ORDER if m in left.index and m in right.index]
    common_assets = [c for c in left.columns if c in right.columns]
    left = left.loc[common_models, common_assets]
    right = right.loc[common_models, common_assets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    left.plot(kind="bar", stacked=True, ax=axes[0], legend=False, color=COLORS)
    right.plot(kind="bar", stacked=True, ax=axes[1], legend=False, color=COLORS)
    
    axes[0].set_title("Pesos: Estimación Estática (Single Path)")
    axes[1].set_title("Pesos: Estimación Robusta (López de Prado)")
    
    for ax in axes:
        ax.set_xlabel("Modelo Base")
        ax.set_ylabel("Peso")
        ax.tick_params(axis='x', rotation=0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Activo", bbox_to_anchor=(1.0, 0.5), loc="center left", frameon=True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_lp_model_summary(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df.copy()
    plot_df["base_model"] = pd.Categorical(plot_df["base_model"], categories=BASE_MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("base_model")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_df.plot(kind="bar", x="base_model", y="mean_sharpe_oos", legend=False, ax=axes[0], color=COLORS[0])
    axes[0].set_title("Sharpe Ratio Out-of-Sample Medio")
    axes[0].set_xlabel("Modelo Base")
    axes[0].set_ylabel("Ratio de Sharpe")
    axes[0].tick_params(axis='x', rotation=0)

    ycol = "alpha_center" if "alpha_center" in plot_df.columns else "robust_score_raw"
    plot_df.plot(kind="bar", x="base_model", y=ycol, legend=False, ax=axes[1], color=COLORS[1])
    axes[1].set_title("Ponderación Central Sugerida (Robustez)")
    axes[1].set_xlabel("Modelo Base")
    axes[1].set_ylabel("Score de Robustez")
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_oos_sharpe_distribution(df: pd.DataFrame, outpath: Path) -> None:
    models = [m for m in BASE_MODEL_ORDER if m in df["base_model"].unique()]
    series = [df.loc[df["base_model"] == m, "sharpe_oos"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(series, tick_labels=models, showfliers=False, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title("Distribución de Sharpe OOS por Escenarios Sintéticos")
    ax.set_xlabel("Modelo Base")
    ax.set_ylabel("Ratio de Sharpe OOS")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_meta_cloud_and_frontier(all_portfolios: pd.DataFrame, frontier: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if "robust_score" in all_portfolios.columns:
        sc = ax.scatter(
            all_portfolios["volatility"],
            all_portfolios["expected_return"],
            c=all_portfolios["robust_score"],
            cmap="viridis",
            s=8,
            alpha=0.25,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Score de Robustez (López de Prado)")
    else:
        ax.scatter(all_portfolios["volatility"], all_portfolios["expected_return"], 
                   s=6, alpha=0.2, label="Meta-portafolios (MC)", color='gray')
    
    ax.plot(frontier["volatility"], frontier["expected_return"], 
            color="#C44E52", linewidth=2.5, label="Frontera Eficiente Robusta")
    
    ax.set_title("Meta-Modelo: Espacio de Soluciones y Frontera Robusta")
    ax.set_xlabel("Volatilidad Esperada (Anualizada)")
    ax.set_ylabel("Retorno Esperado (Anualizado)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_frontier_profiles(frontier_sampled: pd.DataFrame, selected: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_sampled["volatility"], frontier_sampled["expected_return"], 
            marker="o", markersize=3, linewidth=1, color=COLORS[0], alpha=0.5, label="Frontera Sampleada")
    
    ax.scatter(selected["volatility"], selected["expected_return"], 
               s=100, marker="X", color="#C44E52", label="Puntos de Perfiles Seleccionados", zorder=5)
    
    for _, row in selected.iterrows():
        ax.annotate(str(row["profile_bucket"]).replace("_", " ").title(), 
                    (row["volatility"], row["expected_return"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
    
    ax.set_title("Asignación de Perfiles sobre la Frontera Eficiente")
    ax.set_xlabel("Volatilidad")
    ax.set_ylabel("Retorno")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_backtest_growth(monthly: pd.DataFrame, outpath: Path) -> None:
    date_col = "date" if "date" in monthly.columns else monthly.columns[0]
    df = monthly.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    pivot = df.pivot(index=date_col, columns="profile_bucket", values="portfolio_value")
    
    ordered = [c for c in PROFILE_ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in PROFILE_ORDER]
    pivot = pivot[ordered]
    
    ax = pivot.plot(figsize=(10, 6), linewidth=2, alpha=0.85)
    ax.set_title("Backtest: Evolución de la Inversión por Perfil de Riesgo")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Valor Cuotaparte (Base 100)")
    ax.legend(title="Perfil", frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_rebalances(summary: pd.DataFrame, outpath: Path) -> None:
    df = _set_profile_order(summary)
    ax = df.plot(kind="bar", x="profile_bucket", y="n_rebalances", legend=False, figsize=(9, 5), color=COLORS[2])
    ax.set_title("Frecuencia de Rebalanceos por Perfil")
    ax.set_xlabel("Perfil")
    ax.set_ylabel("Cantidad de Operaciones")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    
def plot_backtest_risk_return(summary: pd.DataFrame, outpath: Path) -> None:
    """Genera un gráfico de dispersión de Riesgo vs Retorno para los perfiles backtesteados."""
    df = _set_profile_order(summary)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Graficar puntos con estilo mejorado
    ax.scatter(df["annual_volatility_backtest"], df["annual_return_backtest"], 
               s=100, color=COLORS[0], edgecolors='white', linewidth=1.5, zorder=3, label="Perfiles")
    
    # Añadir etiquetas a cada punto (nombres de perfiles)
    for _, row in df.iterrows():
        ax.annotate(str(row["profile_bucket"]).replace("_", " ").title(), 
                    (row["annual_volatility_backtest"], row["annual_return_backtest"]),
                    xytext=(7, 7), textcoords='offset points', fontsize=9, weight='medium')
    
    ax.set_title("Resultados del Backtest: Relación Riesgo-Retorno por Perfil", pad=20)
    ax.set_xlabel("Volatilidad Anualizada (Riesgo)")
    ax.set_ylabel("Retorno Anualizado (Rendimiento)")
    
    # Ajustar límites para que las etiquetas no queden cortadas
    ax.margins(0.15)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_turnover(summary: pd.DataFrame, outpath: Path) -> None:
    if "annual_turnover_average" not in summary.columns:
        return
    df = _set_profile_order(summary)
    ax = df.plot(kind="bar", x="profile_bucket", y="annual_turnover_average", legend=False, figsize=(9, 5), color=COLORS[3])
    ax.set_title("Rotación de Cartera (Turnover) Anual Promedio")
    ax.set_xlabel("Perfil")
    ax.set_ylabel("Tasa de Rotación Anual")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_selected_profiles_weights(selected: pd.DataFrame, outpath: Path) -> None:
    plot_df = _set_profile_order(selected)[["profile_bucket"] + _asset_weight_columns(selected)].copy()
    plot_df = plot_df.set_index("profile_bucket")
    plot_df.columns = [c.replace("w_", "") for c in plot_df.columns]
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(10, 6), color=COLORS)
    ax.set_title("Composición de Activos por Perfil", pad=20)
    ax.set_xlabel("Perfil de Riesgo")
    ax.set_ylabel("Peso Relativo (%)")
    ax.legend(title="Activo", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_selected_profiles_alphas(selected: pd.DataFrame, outpath: Path) -> None:
    alpha_cols = _alpha_columns(selected)
    if not alpha_cols:
        return
    plot_df = _set_profile_order(selected)[["profile_bucket"] + alpha_cols].copy()
    plot_df = plot_df.set_index("profile_bucket")
    plot_df.columns = [c.replace("alpha_", "").upper() for c in plot_df.columns]
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(10, 6), color=COLORS)
    ax.set_title("Influencia de Modelos Base en la Selección de Meta-Carteras", pad=20)
    ax.set_xlabel("Perfil de Riesgo")
    ax.set_ylabel("Ponderación del Modelo")
    ax.legend(title="Modelo Base", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gráficos para la tesis usando outputs del modelo robusto estilo López de Prado.")
    parser.add_argument("--output-dir", required=True, help="Carpeta con los CSV del modelo robusto")
    parser.add_argument("--charts-dir", default=None, help="Carpeta destino de gráficos. Si se omite, usa output-dir/charts_lp")
    parser.add_argument("--baseline-dir", default=None, help="Carpeta con outputs de la versión anterior para comparar asignaciones")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    charts_dir = Path(args.charts_dir) if args.charts_dir else output_dir / "charts_lp"
    _ensure_dir(charts_dir)

    bundle = _load_bundle(output_dir)

    if bundle["base_robust"] is not None:
        plot_base_model_weights(bundle["base_robust"], charts_dir / "01_modelos_base_robustos.png", "Composición de activos por modelo base (Estimación Robusta)")

    if bundle["base_single"] is not None and bundle["base_robust"] is not None:
        plot_base_model_weights_comparison(bundle["base_single"], bundle["base_robust"], charts_dir / "02_modelos_base_single_vs_robustos.png")

    if bundle["lp_summary"] is not None:
        plot_lp_model_summary(bundle["lp_summary"], charts_dir / "03_lp_resumen_robustez_modelos.png")

    if bundle["scenario_oos"] is not None:
        plot_oos_sharpe_distribution(bundle["scenario_oos"], charts_dir / "04_lp_distribucion_sharpe_oos.png")

    if bundle["all_meta"] is not None and bundle["frontier_full"] is not None:
        plot_meta_cloud_and_frontier(bundle["all_meta"], bundle["frontier_full"], charts_dir / "05_nube_y_frontera_robusta.png")

    if bundle["frontier_sampled"] is not None and bundle["selected"] is not None:
        plot_frontier_profiles(bundle["frontier_sampled"], bundle["selected"], charts_dir / "06_perfiles_en_frontera.png")
        plot_selected_profiles_weights(bundle["selected"], charts_dir / "07_pesos_por_perfil.png")
        plot_selected_profiles_alphas(bundle["selected"], charts_dir / "08_alphas_por_perfil.png")

    if bundle["monthly"] is not None and bundle["summary"] is not None:
        plot_backtest_growth(bundle["monthly"], charts_dir / "13_evolucion_valor_cartera.png")
        plot_rebalances(bundle["summary"], charts_dir / "15_rebalanceos_por_perfil.png")
        plot_turnover(bundle["summary"], charts_dir / "16_turnover_anual_promedio.png")
        plot_backtest_risk_return(bundle["summary"], charts_dir / "14_backtest_riesgo_retorno.png")

if __name__ == "__main__":
    main()
