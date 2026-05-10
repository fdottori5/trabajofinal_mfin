import argparse
from pathlib import Path
import os

os.environ.pop('MPLBACKEND', None)
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE_ORDER = [
    'muy_conservador',
    'conservador',
    'moderado',
    'levemente_riesgoso',
    'muy_riesgoso',
]

ALT_ORDER = [
    'Plazo fijo UVA',
    'Plazo fijo BADLAR',
    'Money market',
    'Merval',
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "font.family": "serif",  
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "medium",
    "legend.frameon": True,
    "legend.fontsize": 10,
    "legend.edgecolor": "#cccccc",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300
})

# Paleta de colores profesional (Muted & Distinguishable)
COLOR_MODEL = "#C44E52"  # Rojo académico para tus perfiles
COLOR_BENCH = ["#4C72B0", "#55A868", "#8172B3", "#937860"] # Azules, verdes y grises para benchmarks
PROFILE_ORDER = ["muy_conservador", "conservador", "moderado", "levemente_riesgoso", "muy_riesgoso"]
ARG_BENCHMARKS = ["Dolar MEP", "Plazo Fijo UVA USD", "FCI MM USD"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Construye la comparacion modelo vs alternativas argentinas desde archivos raw.')
    p.add_argument('--model-monthly', required=True, help='Ruta a 26_profile_backtest_monthly.csv')
    p.add_argument('--argentina-csv', required=True, help='Ruta al CSV con FECHA, MERVAL, UVA, BADLAR, MPFONDO, MEP')
    p.add_argument('--output-dir', required=True, help='Directorio de salida')
    return p.parse_args()


def parse_bbg_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return np.nan
    s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_argentina_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ['FECHA', 'MERVAL', 'UVA', 'BADLAR', 'MPFONDO', 'MEP']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas en el CSV argentino: {missing}')
    df = df[expected].copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True)
    for c in expected[1:]:
        df[c] = df[c].apply(parse_bbg_number)
    df = df.sort_values('FECHA').set_index('FECHA').ffill()
    return df


def load_model_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ['date', 'profile_bucket', 'portfolio_return', 'portfolio_value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas en el backtest mensual: {missing}')
    df = df[required].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['profile_bucket'] = df['profile_bucket'].astype(str)
    return df.sort_values(['profile_bucket', 'date'])


def previous_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts.to_period('M') - 1).to_timestamp('M')


def reconstruct_model_curves(model_monthly: pd.DataFrame) -> pd.DataFrame:
    curves = []
    for profile, grp in model_monthly.groupby('profile_bucket', sort=False):
        grp = grp.sort_values('date').copy()
        first = grp.iloc[0]
        init_date = previous_month_end(first['date'])
        init_value = first['portfolio_value'] / (1.0 + first['portfolio_return'])
        init_row = pd.DataFrame({'date': [init_date], 'profile_bucket': [profile], 'portfolio_value': [init_value]})
        cur = pd.concat([init_row, grp[['date', 'profile_bucket', 'portfolio_value']]], ignore_index=True)
        curves.append(cur)
    out = pd.concat(curves, ignore_index=True)
    wide = out.pivot(index='date', columns='profile_bucket', values='portfolio_value').sort_index()
    ordered = [p for p in PROFILE_ORDER if p in wide.columns] + [c for c in wide.columns if c not in PROFILE_ORDER]
    return wide[ordered]



def simulate_uva_rolling(daily_uva: pd.Series, term_days: int = 180, annual_spread: float = 0.01, base: float = 100.0) -> pd.Series:
    s = daily_uva.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    idx = s.index
    values = pd.Series(index=idx, dtype=float)
    principal = base
    term_start = idx[0]
    uva_start = float(s.loc[term_start])
    maturity = term_start + pd.Timedelta(days=term_days)
    for dt in idx:
        while dt >= maturity:
            pos = idx.searchsorted(maturity, side='right') - 1
            pos = max(pos, 0)
            mdate = idx[pos]
            elapsed = (mdate - term_start).days
            principal = principal * (float(s.loc[mdate]) / uva_start) * ((1.0 + annual_spread) ** (elapsed / 365.0))
            term_start = mdate
            uva_start = float(s.loc[term_start])
            maturity = term_start + pd.Timedelta(days=term_days)
        elapsed = (dt - term_start).days
        values.loc[dt] = principal * (float(s.loc[dt]) / uva_start) * ((1.0 + annual_spread) ** (elapsed / 365.0))
    return values



def simulate_badlar_rolling(daily_rate_pct: pd.Series, term_days: int = 30, base: float = 100.0) -> pd.Series:
    s = daily_rate_pct.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    idx = s.index
    values = pd.Series(index=idx, dtype=float)
    principal = base
    term_start = idx[0]
    rate_start = float(s.loc[term_start]) / 100.0
    maturity = term_start + pd.Timedelta(days=term_days)
    for dt in idx:
        while dt >= maturity:
            pos = idx.searchsorted(maturity, side='right') - 1
            pos = max(pos, 0)
            mdate = idx[pos]
            elapsed = (mdate - term_start).days
            principal = principal * ((1.0 + rate_start) ** (elapsed / 365.0))
            term_start = mdate
            rate_start = float(s.loc[term_start]) / 100.0
            maturity = term_start + pd.Timedelta(days=term_days)
        elapsed = (dt - term_start).days
        values.loc[dt] = principal * ((1.0 + rate_start) ** (elapsed / 365.0))
    return values



def asof_on_dates(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    s = series.dropna().sort_index()
    out = []
    for d in dates:
        pos = s.index.searchsorted(pd.Timestamp(d), side='right') - 1
        if pos < 0:
            out.append(np.nan)
        else:
            out.append(float(s.iloc[pos]))
    return pd.Series(out, index=dates)



def build_alternative_curves(arg_daily: pd.DataFrame, model_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    uva_daily = simulate_uva_rolling(arg_daily['UVA'])
    badlar_daily = simulate_badlar_rolling(arg_daily['BADLAR'])

    curves_ars = pd.DataFrame({
        'Plazo fijo UVA': asof_on_dates(uva_daily, model_dates),
        'Plazo fijo BADLAR': asof_on_dates(badlar_daily, model_dates),
        'Money market': asof_on_dates(arg_daily['MPFONDO'], model_dates),
        'Merval': asof_on_dates(arg_daily['MERVAL'], model_dates),
    })

    mep = asof_on_dates(arg_daily['MEP'], model_dates)
    curves_usd = curves_ars.div(mep, axis=0)

    curves_ars = curves_ars.div(curves_ars.iloc[0]).mul(100.0)
    curves_usd = curves_usd.div(curves_usd.iloc[0]).mul(100.0)

    return curves_ars, curves_usd



def annualized_metrics(levels: pd.Series) -> dict:
    levels = levels.dropna().astype(float)
    rets = levels.pct_change().dropna()
    n = len(rets)
    if len(levels) < 2 or n == 0:
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'final_value': np.nan,
        }
    total_return = levels.iloc[-1] / levels.iloc[0] - 1.0
    annual_return = (levels.iloc[-1] / levels.iloc[0]) ** (12.0 / n) - 1.0
    annual_volatility = rets.std(ddof=1) * np.sqrt(12.0)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    max_drawdown = (levels / levels.cummax() - 1.0).min()
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'final_value': levels.iloc[-1],
    }



def make_metrics(model_curves: pd.DataFrame, alt_curves: pd.DataFrame, currency_label: str) -> pd.DataFrame:
    rows = []
    for col in model_curves.columns:
        m = annualized_metrics(model_curves[col])
        rows.append({'strategy': col, 'type': 'Modelo', 'currency': currency_label, **m})
    for col in alt_curves.columns:
        m = annualized_metrics(alt_curves[col])
        rows.append({'strategy': col, 'type': 'Argentina', 'currency': currency_label, **m})
    df = pd.DataFrame(rows)
    order = PROFILE_ORDER + ALT_ORDER
    df['strategy'] = pd.Categorical(df['strategy'], categories=order, ordered=True)
    df = df.sort_values(['type', 'strategy']).reset_index(drop=True)
    return df



def fmt_pct(x: float) -> str:
    return '' if pd.isna(x) else f'{x*100:.2f}%'


def fmt_num(x: float) -> str:
    return '' if pd.isna(x) else f'{x:.2f}'



def plot_curves_usd(curves: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ordered = [c for c in PROFILE_ORDER if c in curves.columns] + [c for c in ALT_ORDER if c in curves.columns]
    for col in ordered:
        ax.plot(curves.index, curves[col], linewidth=2 if col in PROFILE_ORDER else 1.8, label=col)
    ax.set_title('Evolucion acumulada en USD MEP (base 100)')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Indice base 100')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



def plot_scatter(metrics_usd: pd.DataFrame, outpath: Path) -> None:
    df = metrics_usd.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for kind in ['Modelo', 'Argentina']:
        sub = df[df['type'] == kind]
        ax.scatter(sub['annual_volatility'] * 100, sub['annual_return'] * 100, s=70, label=kind)
        for _, row in sub.iterrows():
            ax.annotate(str(row['strategy']), (row['annual_volatility'] * 100, row['annual_return'] * 100), fontsize=8)
    ax.set_title('Comparacion riesgo-retorno anualizado en USD MEP')
    ax.set_xlabel('Volatilidad anualizada (%)')
    ax.set_ylabel('Retorno anualizado (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



def plot_drawdown(metrics_usd: pd.DataFrame, outpath: Path) -> None:
    df = metrics_usd.copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df['strategy'].astype(str), df['max_drawdown'] * 100)
    ax.set_title('Drawdown maximo comparado en USD MEP')
    ax.set_ylabel('Drawdown maximo (%)')
    ax.tick_params(axis='x', rotation=35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



def plot_final_value(metrics_usd: pd.DataFrame, outpath: Path) -> None:
    df = metrics_usd.copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df['strategy'].astype(str), df['final_value'])
    ax.set_title('Valor final base 100 en USD MEP')
    ax.set_ylabel('Valor final')
    ax.tick_params(axis='x', rotation=35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



def render_table(metrics_usd: pd.DataFrame, out_png: Path, out_xlsx: Path, out_csv: Path) -> None:
    table = metrics_usd.copy()
    table = table[['strategy', 'type', 'annual_return', 'annual_volatility', 'sharpe', 'max_drawdown', 'final_value']]
    table = table.rename(columns={
        'strategy': 'Estrategia',
        'type': 'Tipo',
        'annual_return': 'Retorno anualizado',
        'annual_volatility': 'Volatilidad anualizada',
        'sharpe': 'Sharpe',
        'max_drawdown': 'Drawdown maximo',
        'final_value': 'Valor final (base 100)',
    })
    table.to_csv(out_csv, index=False)
    table.to_excel(out_xlsx, index=False)

    display = table.copy()
    for c in ['Retorno anualizado', 'Volatilidad anualizada', 'Drawdown maximo']:
        display[c] = display[c].apply(fmt_pct)
    for c in ['Sharpe', 'Valor final (base 100)']:
        display[c] = display[c].apply(fmt_num)

    n_rows = len(display) + 1
    fig_h = max(2.5, 0.45 * n_rows)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=display.values, colLabels=display.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()



def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_monthly = load_model_monthly(args.model_monthly)
    model_curves = reconstruct_model_curves(model_monthly)

    arg_daily = load_argentina_daily(args.argentina_csv)
    model_dates = model_curves.index
    alts_ars, alts_usd = build_alternative_curves(arg_daily, model_dates)

    metrics_ars = make_metrics(model_curves.mul(asof_on_dates(arg_daily['MEP'], model_dates).div(asof_on_dates(arg_daily['MEP'], model_dates).iloc[0]), axis=0), alts_ars, 'ARS')
    metrics_usd = make_metrics(model_curves, alts_usd, 'USD MEP')

    combined_usd_curves = pd.concat([model_curves, alts_usd], axis=1)
    combined_ars_curves = pd.concat([
        model_curves.mul(asof_on_dates(arg_daily['MEP'], model_dates).div(asof_on_dates(arg_daily['MEP'], model_dates).iloc[0]), axis=0),
        alts_ars,
    ], axis=1)

    metrics_ars.to_csv(outdir / '01_metricas_ars.csv', index=False)
    metrics_usd.to_csv(outdir / '02_metricas_usd.csv', index=False)
    combined_ars_curves.to_csv(outdir / '03_curvas_mensuales_ars.csv', index_label='date')
    combined_usd_curves.to_csv(outdir / '04_curvas_mensuales_usd.csv', index_label='date')

    plot_curves_usd(combined_usd_curves, outdir / '05_curvas_acumuladas_usd.png')
    plot_scatter(metrics_usd, outdir / '06_scatter_riesgo_retorno_usd.png')
    plot_drawdown(metrics_usd, outdir / '07_drawdown_maximo_usd.png')
    plot_final_value(metrics_usd, outdir / '08_valor_final_base100_usd.png')
    render_table(metrics_usd, outdir / '09_tabla_metricas_comparativas.png', outdir / '10_tabla_metricas_comparativas.xlsx', outdir / '11_tabla_metricas_comparativas.csv')

    print(f'[OK] Archivos generados en: {outdir}')


if __name__ == '__main__':
    main()
