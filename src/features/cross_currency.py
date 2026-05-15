"""
src/features/cross_currency.py
==============================
Cross-currency correlation and synthetic index features.

These capture macro FX structure (e.g. broad USD strength)
that per-pair models cannot see on their own.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


def add_cross_currency_features(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Add cross-currency features to the wide (pre-encoding) DataFrame.

    Features added per currency_code row:
      - corr_vs_base_{window}d  : rolling {window}-day Pearson correlation
                                   of this pair's log_return vs the base pair
      - usd_index               : equal-weight mean of top-N pairs' exchange_rate
                                   (proxy for broad USD direction)

    Parameters
    ----------
    df  : DataFrame with columns [date, currency_code, exchange_rate, log_return, ...]
    cfg : cross_currency section of features.yaml

    Returns
    -------
    DataFrame with new cross-currency columns.
    """
    if not cfg.get("enabled", False):
        return df

    df = df.copy().sort_values(["date", "currency_code"]).reset_index(drop=True)
    corr_window     = cfg.get("corr_window", 30)
    base_currency   = cfg.get("base_currency", None)
    index_currencies = cfg.get("index_currencies", [])

    # ── Pivot to wide: one column per currency ────────────────────────────────
    pivot = df.pivot_table(
        index="date", columns="currency_code", values="exchange_rate", aggfunc="mean"
    )
    log_ret_pivot = np.log(pivot / pivot.shift(1)).fillna(0)

    codes = pivot.columns.tolist()

    # ── Auto-detect top-N currencies for index if not specified ───────────────
    if not index_currencies:
        counts = df["currency_code"].value_counts()
        index_currencies = counts.head(min(5, len(counts))).index.tolist()
        log.info(f"Auto-selected index currencies: {index_currencies}")

    # ── Synthetic currency index (equal-weight mean of selected pairs) ─────────
    available_index = [c for c in index_currencies if c in pivot.columns]
    if available_index:
        usd_index_series = pivot[available_index].mean(axis=1)
    else:
        log.warning("No index currencies found in pivot; skipping USD index.")
        usd_index_series = None

    # ── Rolling correlation vs base pair ─────────────────────────────────────
    if base_currency and base_currency in log_ret_pivot.columns:
        base_returns = log_ret_pivot[base_currency]
        corr_dict = {}
        for code in codes:
            if code == base_currency:
                continue
            rolling_corr = (
                log_ret_pivot[code]
                .rolling(corr_window, min_periods=max(5, corr_window // 3))
                .corr(base_returns)
                .fillna(0)
            )
            corr_dict[code] = rolling_corr
        corr_df = pd.DataFrame(corr_dict)  # index = date
    else:
        log.warning(
            f"Base currency '{base_currency}' not found or not set; skipping correlation features."
        )
        corr_df = None

    # ── Merge back into long format ───────────────────────────────────────────
    rows = []
    for code, grp in df.groupby("currency_code"):
        grp = grp.copy().set_index("date")

        if corr_df is not None and code in corr_df.columns:
            grp[f"corr_vs_{base_currency}_{corr_window}d"] = corr_df[code].reindex(grp.index).fillna(0)
        else:
            grp[f"corr_vs_{base_currency}_{corr_window}d"] = 0.0

        if usd_index_series is not None:
            grp["fx_index"] = usd_index_series.reindex(grp.index).fillna(method="ffill")
            # Relative strength: how this pair moves vs the index
            grp["rate_vs_index"] = grp["exchange_rate"] / (grp["fx_index"] + 1e-10)
        else:
            grp["fx_index"]      = 0.0
            grp["rate_vs_index"] = 0.0

        rows.append(grp.reset_index())

    out = pd.concat(rows, ignore_index=True)
    log.info(
        f"Cross-currency features added: "
        f"corr_vs_{base_currency}_{corr_window}d, fx_index, rate_vs_index"
    )
    return out
