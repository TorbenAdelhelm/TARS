"""Probability-based intraday research policy. No brokerage connection or orders.

Run `python trading_policy.py --help` and read README.md before using results.
All timestamps are UTC. Yahoo intraday timestamps denote the START of a bar.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits


CLASS_NAMES = ["unfavourable", "neutral", "favourable"]
TRADE_COLUMNS = ["symbol", "signal_bar", "signal_available_at", "planned_entry_at",
                 "planned_exit_at", "entry_price", "exit_price", "entry_fill", "budget",
                 "p_favourable", "expected_net_bps", "exit_fill", "net_return", "portfolio_pnl"]
BASE_FEATURES = [
    "return_1", "return_3", "return_6", "return_12", "volatility_12",
    "range_fraction", "body_fraction", "volume_ratio", "vwap_distance",
    "session_return", "time_sin", "time_cos",
]


@dataclass
class Config:
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    interval_minutes: int = 5
    history_days: int = 59
    holding_bars: int = 3
    entry_delay_bars: int = 1  # after signal availability; entry at i+2
    cost_bps_per_side: float = 2.0  # spread, slippage AND fees, assumed
    neutral_buffer_bps: float = 1.0
    max_position_weight: float = 0.10
    max_gross_weight: float = 0.30
    min_validation_trades: int = 8
    seed: int = 42

    def validate(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Supply a nonempty list of distinct symbols.")
        if any(not re.fullmatch(r"[A-Z0-9.^=-]+", s) for s in self.symbols):
            raise ValueError("Unsupported symbol spelling.")
        if self.interval_minutes not in (1, 2, 5, 15, 30):
            raise ValueError("Supported bar lengths: 1, 2, 5, 15, 30 minutes.")
        if not 1 <= self.history_days <= 59:
            raise ValueError("history_days must be between 1 and 59.")
        if self.holding_bars < 1 or self.entry_delay_bars < 1:
            raise ValueError("Holding horizon and execution delay must be positive.")
        if not 0 < self.max_position_weight <= self.max_gross_weight <= 1:
            raise ValueError("Require 0 < position weight <= gross weight <= 1.")
        if not 0 <= self.cost_bps_per_side < 10_000 or self.neutral_buffer_bps < 0:
            raise ValueError("Require 0 <= per-side cost < 10000 bps and a nonnegative buffer.")
        if self.min_validation_trades < 1:
            raise ValueError("min_validation_trades must be positive.")

    @property
    def bar(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.interval_minutes)

    @property
    def cost(self) -> float:
        return self.cost_bps_per_side / 10_000

    @property
    def label_hurdle(self) -> float:
        # Exact break-even raw return with symmetric proportional execution costs.
        return (1 + self.cost) / (1 - self.cost) - 1 + self.neutral_buffer_bps / 10_000


def utc(value: Any | None = None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Timestamps must include a timezone, e.g. 2026-10-15T15:00:00Z.")
    return t.tz_convert("UTC")


def write_json(path: Path, value: Any) -> None:
    def convert(v: Any) -> Any:
        if isinstance(v, (pd.Timestamp, pd.Timedelta, Path)):
            return str(v)
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def clean_bars(frame: pd.DataFrame, symbol: str, cfg: Config, now: pd.Timestamp) -> pd.DataFrame:
    """Validate OHLCV; filter to completed, regular US-equity sessions.

    This example deliberately supports instruments with NYSE regular hours only.
    A different exchange requires a different calendar and feature/session logic.
    """
    f = frame.copy()
    f.columns = [str(c).lower().replace(" ", "_") for c in f.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(f.columns):
        raise ValueError(f"{symbol}: missing OHLCV columns.")
    if not isinstance(f.index, pd.DatetimeIndex) or f.index.tz is None:
        raise ValueError(f"{symbol}: bar index must be timezone-aware.")
    f.index = f.index.tz_convert("UTC")
    f = f[required].apply(pd.to_numeric, errors="raise").sort_index()
    if f.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate bar timestamps.")
    f = f.loc[f.index + cfg.bar <= now].copy()
    if f.empty:
        raise ValueError(f"{symbol}: no completed bars.")
    if not np.isfinite(f.to_numpy()).all():
        raise ValueError(f"{symbol}: missing/nonfinite OHLCV values; repair the data first.")
    prices = f[["open", "high", "low", "close"]]
    bad = (prices <= 0).any(axis=1) | (f.volume < 0)
    bad |= f.high < prices.max(axis=1)
    bad |= f.low > prices.min(axis=1)
    if bad.any():
        raise ValueError(f"{symbol}: invalid OHLCV values.")
    dates = f.index.tz_convert("America/New_York").date.astype(str)
    schedule = mcal.get_calendar("NYSE").schedule(start_date=str(min(dates)), end_date=str(max(dates)))
    schedule.index = schedule.index.strftime("%Y-%m-%d")
    f["session"] = dates
    f["session_open"] = pd.to_datetime(f.session.map(schedule.market_open), utc=True)
    f["session_close"] = pd.to_datetime(f.session.map(schedule.market_close), utc=True)
    f = f.loc[(f.index >= f.session_open) & (f.index + cfg.bar <= f.session_close)].copy()
    if f.empty:
        raise ValueError(f"{symbol}: no bars within NYSE regular trading hours.")
    if (((f.index - f.session_open) / cfg.bar) % 1 != 0).any():
        raise ValueError(f"{symbol}: timestamps are not aligned to the chosen interval.")
    f["symbol"] = symbol
    f.index.name = "timestamp"
    return f


def load_data(cfg: Config, now: pd.Timestamp, data_dir: Path | None = None,
              save_dir: Path | None = None,
              cache_dir: Path = Path("work/yfinance_cache")) -> tuple[dict[str, pd.DataFrame], dict]:
    """Download all requested symbols or read matching timestamp/open/... CSVs.

    Fail if any symbol is unavailable; never silently change the universe.
    """
    frames, metadata = {}, {"source": "csv" if data_dir else "Yahoo via yfinance",
                            "downloaded_at": now, "symbols": {}}
    for symbol in cfg.symbols:
        print(f"Loading {symbol} ...", file=sys.stderr, flush=True)
        if data_dir:
            raw = pd.read_csv(data_dir / f"{symbol}.csv")
            if "timestamp" not in raw:
                raise ValueError(f"{symbol}.csv needs a timestamp column.")
            # Require explicit timezones; do not silently interpret naive local time.
            if not raw.timestamp.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
                raise ValueError("CSV timestamps require a UTC offset or Z suffix.")
            raw.index = pd.to_datetime(raw.pop("timestamp"), utc=True)
        else:
            import yfinance as yf
            cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_dir.resolve()))
            # Use date boundaries; Yahoo end is exclusive. Restrict completed bars below.
            start = (now - pd.Timedelta(days=cfg.history_days)).date().isoformat()
            end = (now + pd.Timedelta(days=1)).date().isoformat()
            raw = yf.download(symbol, start=start, end=end,
                              interval=f"{cfg.interval_minutes}m", auto_adjust=True,
                              prepost=False, actions=False, progress=False, threads=False,
                              ignore_tz=False, multi_level_index=False, timeout=30)
            if raw.empty:
                raise RuntimeError(f"Yahoo returned no bars for {symbol}. Check access, symbol and intraday history limits.")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
        bars = clean_bars(raw, symbol, cfg, now)
        frames[symbol] = bars
        metadata["symbols"][symbol] = {"bars": len(bars), "first_bar": bars.index[0],
                                        "last_completed_bar": bars.index[-1],
                                        "sessions": int(bars.session.nunique())}
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            bars[["open", "high", "low", "close", "volume"]].to_csv(save_dir / f"{symbol}.csv")
    return frames, metadata


def make_features(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Past/current information only; reset at session starts and after gaps."""
    f = bars.copy()
    # A missing bar starts a new segment: 12-bar features cannot span an outage.
    new_segment = f.session.ne(f.session.shift()) | f.index.to_series().diff().ne(cfg.bar)
    segment = new_segment.cumsum()
    for _, s in f.groupby(segment, sort=False):
        idx = s.index
        for lag in (1, 3, 6, 12):
            f.loc[idx, f"return_{lag}"] = np.log(s.close / s.close.shift(lag))
        returns = np.log(s.close / s.close.shift())
        f.loc[idx, "volatility_12"] = returns.rolling(12, min_periods=12).std(ddof=1)
        baseline = s.volume.shift().rolling(12, min_periods=12).mean()
        f.loc[idx, "volume_ratio"] = s.volume / baseline.replace(0, np.nan)
    f["range_fraction"] = (f.high - f.low) / f.close
    f["body_fraction"] = (f.close - f.open) / f.open
    typical = (f.high + f.low + f.close) / 3
    cum_pv = (typical * f.volume).groupby(f.session).cumsum()
    cum_v = f.volume.groupby(f.session).cumsum().replace(0, np.nan)
    f["vwap_distance"] = f.close / (cum_pv / cum_v) - 1
    first_open = f.groupby("session").open.transform("first")
    f["session_return"] = f.close / first_open - 1
    elapsed = (f.index + cfg.bar - f.session_open).dt.total_seconds()
    duration = (f.session_close - f.session_open).dt.total_seconds()
    f["time_sin"] = np.sin(2 * np.pi * elapsed / duration)
    f["time_cos"] = np.cos(2 * np.pi * elapsed / duration)
    for symbol in cfg.symbols:
        f[f"instrument_{symbol}"] = (f.symbol == symbol).astype(float)
    f["signal_available_at"] = f.index + cfg.bar
    f["planned_entry_at"] = f.index + (1 + cfg.entry_delay_bars) * cfg.bar
    f["planned_exit_at"] = f.planned_entry_at + cfg.holding_bars * cfg.bar
    return f


def feature_names(cfg: Config) -> list[str]:
    return BASE_FEATURES + [f"instrument_{s}" for s in cfg.symbols]


def make_labels(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Entry and exit at future opens, with complete same-session bar paths."""
    f = features.copy()
    delay = 1 + cfg.entry_delay_bars
    end = delay + cfg.holding_bars
    timestamps = f.index.to_series()
    valid = pd.Series(True, index=f.index)
    for offset in range(1, end + 1):
        valid &= timestamps.shift(-offset).eq(timestamps + offset * cfg.bar)
        valid &= f.session.shift(-offset).eq(f.session)
    f["entry_price"] = f.open.shift(-delay)
    f["exit_price"] = f.open.shift(-end)
    f["gross_return"] = f.exit_price / f.entry_price - 1
    f["target"] = np.select([f.gross_return < -cfg.label_hurdle,
                              f.gross_return > cfg.label_hurdle], [0, 2], default=1)
    finite = np.isfinite(f[feature_names(cfg)].to_numpy()).all(axis=1)
    # Exit uses an observed bar OPEN, so it must precede the session close.
    valid &= f.planned_exit_at < f.session_close
    return f.loc[valid & finite].copy()


def chronological_blocks(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Four GLOBAL session blocks shared by every instrument; no random split."""
    sessions = sorted(data.session.unique())
    n = len(sessions)
    if n < 28:
        raise ValueError(f"Only {n} usable sessions; need at least 28. Fetch more history or use saved CSVs.")
    cuts = [0, int(n * .50), int(n * .70), int(n * .85), n]
    blocks, description = {}, {}
    names = ["fit", "calibration", "policy_selection", "test"]
    for i, name in enumerate(names):
        dates = sessions[cuts[i]:cuts[i+1]]
        part = data.loc[data.session.isin(dates)].copy()
        if i < len(names) - 1:
            next_start = pd.Timestamp(str(sessions[cuts[i+1]]), tz="America/New_York").tz_convert("UTC")
            part = part.loc[part.planned_exit_at < next_start]
        if part.empty:
            raise ValueError(f"Empty {name} block.")
        blocks[name] = part
        description[name] = {"first_session": dates[0], "last_session": dates[-1],
                             "sessions": len(dates), "examples": len(part)}
    return blocks, description


def calibrated_probabilities(model: Any, x: pd.DataFrame, temperature: float) -> np.ndarray:
    with threadpool_limits(limits=2, user_api="openmp"):
        p = np.clip(model.predict_proba(x), 1e-12, 1)
    return softmax(np.log(p) / temperature, axis=1)


def predict_rows(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    f = rows.copy()
    p = calibrated_probabilities(bundle["model"], f[bundle["features"]], bundle["temperature"])
    for i, name in enumerate(CLASS_NAMES):
        f[f"p_{name}"] = p[:, i]
    expected_gross = p @ np.asarray(bundle["class_return_means"])
    cfg = Config(**bundle["config"])
    f["expected_net_bps"] = ((1 + expected_gross) * (1 - cfg.cost) / (1 + cfg.cost) - 1) * 10_000
    return f


def eligible(rows: pd.DataFrame, policy: dict) -> pd.Series:
    return ((rows.p_favourable >= policy["min_favourable_probability"])
            & (rows.expected_net_bps >= policy["min_expected_net_bps"]))


def backtest(scored: pd.DataFrame, bars: dict[str, pd.DataFrame], policy: dict,
             cfg: Config) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Unlevered, long-only, event simulation; mark holdings at every bar open.

    Position sizing uses NAV at execution. Entry/exit prices include adverse costs.
    Signals cannot enter while that symbol was held at signal availability.
    """
    sessions = set(scored.session)
    market = pd.concat([b.loc[b.session.isin(sessions)] for b in bars.values()]).sort_index()
    candidates = scored.loc[eligible(scored, policy)].copy()
    candidate_groups = {t: g.sort_values(["expected_net_bps", "symbol"], ascending=[False, True])
                        for t, g in candidates.groupby("signal_available_at")}
    cash = 1.0
    active, pending, marks = {}, {}, {}
    trades, equity = [], []
    peak_gross, peak_position = 0.0, 0.0
    for t, group in market.groupby(level=0, sort=True):
        for _, bar in group.iterrows():
            marks[bar.symbol] = float(bar.open)
        for symbol, pos in list(active.items()):
            if pos["planned_exit_at"] <= t:
                if pos["planned_exit_at"] != t:
                    raise RuntimeError("Missing scheduled exit bar; data integrity failure.")
                proceeds = pos["shares"] * pos["exit_price"] * (1 - cfg.cost)
                cash += proceeds
                trades.append({k: v for k, v in pos.items() if k != "shares"} | {
                    "exit_fill": pos["exit_price"] * (1 - cfg.cost),
                    "net_return": proceeds / pos["budget"] - 1,
                    "portfolio_pnl": proceeds - pos["budget"],
                })
                del active[symbol]
        # Fill only orders decided earlier; recheck cash and risk at execution.
        due = [(s, p) for s, p in pending.items() if p["row"].planned_entry_at <= t]
        for symbol, order in due:
            row, signal_t = order["row"], order["signal_bar"]
            if row.planned_entry_at != t:
                raise RuntimeError("Missing scheduled entry bar; data integrity failure.")
            del pending[symbol]
            gross = sum(p["shares"] * marks[s] for s, p in active.items())
            nav = cash + gross
            budget = min(order["reserved_weight"] * nav, cfg.max_position_weight * nav,
                         max(0.0, cfg.max_gross_weight * nav - gross), cash)
            if budget <= 1e-10:
                continue
            # Budget includes entry cost; raw gross exposure is therefore smaller.
            fill = float(row.entry_price) * (1 + cfg.cost)
            active[symbol] = {
                "symbol": symbol, "signal_bar": signal_t,
                "signal_available_at": row.signal_available_at,
                "planned_entry_at": row.planned_entry_at,
                "planned_exit_at": row.planned_exit_at,
                "entry_price": float(row.entry_price), "exit_price": float(row.exit_price),
                "entry_fill": fill, "budget": budget, "shares": budget / fill,
                "p_favourable": float(row.p_favourable),
                "expected_net_bps": float(row.expected_net_bps),
            }
            cash -= budget
        # A decision reserves capacity against current holdings AND pending entries.
        # It cannot anticipate that a different holding will free cash in the future.
        if t in candidate_groups:
            for signal_t, row in candidate_groups[t].iterrows():
                symbol = row.symbol
                if symbol in active or symbol in pending:
                    continue
                gross = sum(p["shares"] * marks[s] for s, p in active.items())
                nav = cash + gross
                reserved = sum(p["reserved_weight"] for p in pending.values())
                weight = min(cfg.max_position_weight,
                             max(0.0, cfg.max_gross_weight - gross / nav - reserved))
                if weight > 1e-8:
                    pending[symbol] = {"row": row, "signal_bar": signal_t, "reserved_weight": weight}
        values = [p["shares"] * marks[s] for s, p in active.items()]
        nav = cash + sum(values)
        peak_gross = max(peak_gross, sum(values) / nav)
        peak_position = max(peak_position, max(values, default=0.0) / nav)
        equity.append({"timestamp": t, "equity": nav, "cash": cash,
                       "gross_weight": sum(values) / nav})
    if active or pending:
        raise RuntimeError("Unclosed positions/orders at end of test; invalid labels/data.")
    curve = pd.DataFrame(equity)
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    nav = curve.equity
    # Include initial capital when measuring drawdown (first entry may already cost money).
    peaks = np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:]
    dd = float(np.max(1 - nav.to_numpy() / peaks))
    summary = {
        "total_return": float(nav.iloc[-1] - 1), "max_drawdown_bar_open_marks": dd,
        "trades": len(trades), "win_rate": float((trade_frame.net_return > 0).mean()) if trades else None,
        "mean_trade_net_bps": float(trade_frame.net_return.mean() * 10_000) if trades else None,
        "cash_baseline_return": 0.0, "peak_observed_gross_weight": peak_gross,
        "peak_observed_position_weight": peak_position,
        "sessions": len(sessions),
    }
    # Passive comparison: same initial gross allocation, each instrument bought once.
    passive_weight = min(cfg.max_position_weight, cfg.max_gross_weight / len(cfg.symbols))
    passive_return = 0.0
    for symbol, b in bars.items():
        sample = b.loc[b.session.isin(sessions)]
        passive_return += passive_weight * (sample.close.iloc[-1] * (1 - cfg.cost)
                                           / (sample.open.iloc[0] * (1 + cfg.cost)) - 1)
    summary["passive_same_initial_allocation_return"] = float(passive_return)
    return summary, trade_frame, curve


def probability_metrics(bundle: dict, data: pd.DataFrame) -> dict:
    p = calibrated_probabilities(bundle["model"], data[bundle["features"]], bundle["temperature"])
    y = data.target.to_numpy(dtype=int)
    baseline = np.tile(bundle["class_priors"], (len(y), 1))
    onehot = np.eye(3)[y]
    bins = []
    for lo, hi in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        mask = (p[:, 2] >= lo) & (p[:, 2] < min(hi, 1.0000001))
        if mask.any():
            bins.append({"lower": float(lo), "upper": float(hi), "count": int(mask.sum()),
                         "mean_predicted_favourable": float(p[mask, 2].mean()),
                         "observed_favourable_fraction": float((y[mask] == 2).mean())})
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "training_prior_log_loss": float(log_loss(y, baseline, labels=[0, 1, 2])),
            "multiclass_brier_sum": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
            "training_prior_brier_sum": float(np.mean(np.sum((baseline - onehot) ** 2, axis=1))),
            "accuracy": float(np.mean(p.argmax(axis=1) == y)),
            "favourable_reliability_bins": bins}


def train(cfg: Config, bars: dict[str, pd.DataFrame], metadata: dict, out: Path) -> dict:
    cfg.validate()
    out.mkdir(parents=True, exist_ok=True)
    labeled = pd.concat([make_labels(make_features(b, cfg), cfg) for b in bars.values()]).sort_index()
    blocks, split_info = chronological_blocks(labeled)
    for name, block in blocks.items():
        missing = set(cfg.symbols) - set(block.symbol)
        if missing:
            raise ValueError(f"{name} block has no usable examples for {sorted(missing)}.")
    fit, calibration = blocks["fit"], blocks["calibration"]
    counts = fit.target.value_counts().reindex([0, 1, 2], fill_value=0)
    if (counts < 30).any():
        raise ValueError(f"Too few training examples in a class: {counts.to_dict()}. More data or a different horizon is needed.")
    names = feature_names(cfg)
    model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=120,
                max_leaf_nodes=15, max_depth=4, min_samples_leaf=40,
                l2_regularization=5, early_stopping=False, random_state=cfg.seed)
    with threadpool_limits(limits=2, user_api="openmp"):
        model.fit(fit[names], fit.target)
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Training must contain all three outcome classes.")
    calibration_logs = np.log(np.clip(calibrated_probabilities(model, calibration[names], 1.0), 1e-12, 1))
    optimum = minimize_scalar(lambda log_t: log_loss(calibration.target,
                              softmax(calibration_logs / np.exp(log_t), axis=1), labels=[0, 1, 2]),
                              bounds=(np.log(.25), np.log(8.0)), method="bounded")
    if not optimum.success:
        raise RuntimeError("Probability calibration failed.")
    # Means estimated exclusively in the fit block; returns clipped at TRAIN quantiles.
    lower, upper = fit.gross_return.quantile([.01, .99])
    means = fit.assign(clipped=fit.gross_return.clip(lower, upper)).groupby("target").clipped.mean()
    bundle = {"format_version": 1, "model": model, "features": names,
              "temperature": float(np.exp(optimum.x)), "class_return_means": means.reindex([0, 1, 2]).tolist(),
              "class_priors": (counts / counts.sum()).tolist(), "config": asdict(cfg),
              "splits": split_info, "data_metadata": metadata,
              "fitted_through": fit.planned_exit_at.max(),
              "policy_selected_through": blocks["policy_selection"].planned_exit_at.max()}
    validation = predict_rows(bundle, blocks["policy_selection"])
    # Cash is an explicit candidate. Refuse to manufacture a trade recommendation.
    cash_policy = {"min_favourable_probability": 1.01, "min_expected_net_bps": 0.0}
    chosen, best_score = cash_policy, 0.0
    search = []
    for min_p in (.35, .45, .55, .65):
        for min_edge in (0.0, 1.0, 2.0, 4.0, 8.0):
            policy = {"min_favourable_probability": min_p, "min_expected_net_bps": min_edge}
            result, _, _ = backtest(validation, bars, policy, cfg)
            score = result["total_return"] - .5 * result["max_drawdown_bar_open_marks"]
            admissible = result["trades"] >= cfg.min_validation_trades
            search.append(policy | result | {"selection_score": score, "admissible": admissible})
            if admissible and score > best_score:
                chosen, best_score = policy, score
    bundle["policy"] = chosen
    validation_result, _, _ = backtest(validation, bars, chosen, cfg)
    test_scored = predict_rows(bundle, blocks["test"])
    test_result, trades, curve = backtest(test_scored, bars, chosen, cfg)
    report = {
        "purpose": "Research and paper-trading recommendations; no order execution",
        "config": asdict(cfg), "data": metadata, "chronological_splits": split_info,
        "label_hurdle_bps": cfg.label_hurdle * 10_000,
        "temperature": bundle["temperature"],
        "class_gross_return_means_bps": (np.array(bundle["class_return_means"]) * 10_000).tolist(),
        "selected_policy": chosen, "cash_only_policy_selected": chosen == cash_policy,
        "validation_backtest": validation_result, "untouched_test_backtest": test_result,
        "untouched_test_probability_metrics": probability_metrics(bundle, blocks["test"]),
        "policy_candidates_tried": len(search) + 1,
        "limitations": [
            "Recent Yahoo intraday history is short and may be delayed or revised.",
            "Costs are assumptions; there are no bid/ask quotes or realistic queue/fill models.",
            "Probabilities, class return means, and validation-selected thresholds may not generalize.",
            "Bar-open drawdown misses movements within bars; market exposure can drift above entry caps.",
            "Overlapping labels and shared market exposure mean rows/trades are not independent.",
            "This is a single chronological holdout, not a walk-forward profitability study.",
        ],
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "scipy", "yfinance", "pandas-market-calendars")},
    }
    joblib.dump(bundle, out / "model.joblib")
    write_json(out / "report.json", report)
    pd.DataFrame(search).to_csv(out / "policy_search.csv", index=False)
    trades.to_csv(out / "test_trades.csv", index=False)
    curve.to_csv(out / "test_equity.csv", index=False)
    test_scored.reset_index()[["timestamp", "symbol", "target", "gross_return", "p_unfavourable",
                              "p_neutral", "p_favourable", "expected_net_bps"]].to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps({"selected_policy": chosen, "test": test_result}, indent=2), file=sys.stderr)
    return bundle


def read_positions(path: Path | None, cfg: Config) -> dict:
    """Positions are actual holdings, not past recommendations. Nothing is auto-filled."""
    data = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(data, dict) or set(data) - set(cfg.symbols):
        raise ValueError("Position keys must be trained symbols.")
    for symbol, pos in data.items():
        if not isinstance(pos, dict) or not 0 < float(pos.get("weight", 0)) <= 1:
            raise ValueError(f"{symbol}: position needs a positive current NAV weight.")
        pos["exit_at"] = utc(pos["exit_at"])
    if sum(float(p["weight"]) for p in data.values()) > 1:
        raise ValueError("This long-only example does not support leverage.")
    return data


def recommendations(bundle: dict, bars: dict[str, pd.DataFrame], now: pd.Timestamp,
                    positions: dict | None = None) -> dict:
    cfg = Config(**bundle["config"])
    positions = positions or {}
    if "policy_selected_through" in bundle and now <= utc(bundle["policy_selected_through"]):
        raise ValueError("The replay clock predates policy selection; that would use future training information.")
    today = now.tz_convert("America/New_York").date()
    schedule = mcal.get_calendar("NYSE").schedule(start_date=today, end_date=today)
    is_open = bool(not schedule.empty and schedule.market_open.iloc[0] <= now < schedule.market_close.iloc[0])
    results = []
    gross = sum(float(p["weight"]) for p in positions.values())
    proposals = []
    for symbol in cfg.symbols:
        f = make_features(bars[symbol], cfg)
        # Do not fall back to an older feature row if the latest bar has insufficient history.
        latest = f.iloc[[-1]]
        row = latest.iloc[0]
        item = {"symbol": symbol, "bar_started_at": latest.index[0],
                "signal_available_at": row.signal_available_at,
                "data_age_seconds": float((now - row.signal_available_at).total_seconds()),
                "planned_entry_at": row.planned_entry_at, "planned_exit_at": row.planned_exit_at,
                "entry_deadline_exclusive": row.planned_entry_at,
                "current_weight": float(positions.get(symbol, {}).get("weight", 0)),
                "suggested_additional_weight": 0.0, "action": "WAIT", "reason": ""}
        finite = np.isfinite(latest[bundle["features"]].to_numpy()).all()
        if finite:
            scored = predict_rows(bundle, latest).iloc[0]
            item["outcome_probabilities"] = {name: float(scored[f"p_{name}"]) for name in CLASS_NAMES}
            item["estimated_net_return_bps"] = float(scored.expected_net_bps)
            qualifies = bool(eligible(scored.to_frame().T, bundle["policy"]).iloc[0])
        else:
            qualifies = False
            item["outcome_probabilities"] = None
            item["estimated_net_return_bps"] = None
        fresh = row.signal_available_at <= now < row.planned_entry_at
        fits_session = row.planned_exit_at < row.session_close
        if symbol in positions:
            pos = positions[symbol]
            due = now >= pos["exit_at"]
            item["scheduled_exit_at"] = pos["exit_at"]
            if not is_open:
                item.update(action="REVIEW_POSITION", reason="Market is closed; check the holding and its scheduled exit.")
            elif due:
                item.update(action="EXIT_LONG", reason="The recorded fixed holding horizon has elapsed.")
            elif gross > cfg.max_gross_weight or float(pos["weight"]) > cfg.max_position_weight:
                item.update(action="REVIEW_RISK", reason="Current holdings exceed the configured entry allocation limits.")
            else:
                item.update(action="HOLD", reason="Maintain the recorded position until its scheduled exit.")
        elif not is_open:
            item["reason"] = "Outside the current regular trading session."
        elif not fresh:
            item["reason"] = "The data are stale or the planned entry deadline has passed."
        elif not fits_session:
            item["reason"] = "The full holding horizon does not fit before the session close."
        elif not finite:
            item["reason"] = "Insufficient uninterrupted intraday feature history."
        elif not qualifies:
            item["reason"] = ("Validation selected cash; new entries are disabled."
                              if bundle["policy"]["min_favourable_probability"] > 1
                              else "The learned entry thresholds are not met.")
        else:
            proposals.append(item)
        results.append(item)
    # Rank simultaneous opportunities; reserve allocation against ALL existing holdings,
    # including any awaiting exit. Actual sizing must be rechecked at execution.
    for item in sorted(proposals, key=lambda i: (-i["estimated_net_return_bps"], i["symbol"])):
        weight = min(cfg.max_position_weight, max(0.0, cfg.max_gross_weight - gross))
        if weight > 1e-8:
            item.update(action="CONSIDER_OPEN_LONG", suggested_additional_weight=weight,
                        reason="Thresholds met; paper-trade at the planned entry, rechecking holdings and execution costs.")
            gross += weight
        else:
            item["reason"] = "No remaining portfolio allocation."
    return {"generated_at": now, "paper_trading_only": True,
            "model_fitted_through": bundle.get("fitted_through"),
            "policy_selected_through": bundle.get("policy_selected_through"),
            "positions_assumption": "Positions supplied" if positions else "All instruments flat; no positions supplied",
            "probabilities_describe": "Future raw price-return categories, not action probabilities or guaranteed returns",
            "policy": bundle["policy"], "recommendations": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fit = subs.add_parser("train", help="Download, fit, calibrate, select policy and test")
    fit.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"])
    fit.add_argument("--interval-minutes", type=int, default=5)
    fit.add_argument("--history-days", type=int, default=59)
    fit.add_argument("--holding-bars", type=int, default=3)
    fit.add_argument("--cost-bps-per-side", type=float, default=2.0)
    fit.add_argument("--max-position-weight", type=float, default=.10)
    fit.add_argument("--max-gross-weight", type=float, default=.30)
    fit.add_argument("--out", type=Path, default=Path("run"))
    predict = subs.add_parser("recommend", help="Reuse a trusted fitted model; fetch fresh bars")
    predict.add_argument("--model", type=Path, default=Path("run/model.joblib"))
    predict.add_argument("--out", type=Path, default=Path("run"))
    for sub in (fit, predict):
        sub.add_argument("--data-dir", type=Path, help="Use saved timezone-aware OHLCV CSVs instead of Yahoo")
        sub.add_argument("--positions", type=Path, help="JSON of actual current holdings; default: flat")
        sub.add_argument("--as-of", help="Historical replay clock with explicit timezone; never for live use")
    args = parser.parse_args(argv)
    try:
        now = utc(args.as_of)
        args.out.mkdir(parents=True, exist_ok=True)
        if args.command == "train":
            cfg = Config(symbols=tuple(s.upper() for s in args.symbols),
                         interval_minutes=args.interval_minutes, history_days=args.history_days,
                         holding_bars=args.holding_bars, cost_bps_per_side=args.cost_bps_per_side,
                         max_position_weight=args.max_position_weight, max_gross_weight=args.max_gross_weight)
            cfg.validate()
            bars, metadata = load_data(cfg, now, args.data_dir, args.out / "data")
            bundle = train(cfg, bars, metadata, args.out)
        else:
            # Pickle-based model artifacts can execute code. Load only your own trusted files.
            bundle = joblib.load(args.model)
            if bundle.get("format_version") != 1:
                raise ValueError("Unsupported model format.")
            cfg = Config(**bundle["config"])
            cfg.validate()
            bars, _ = load_data(replace(cfg, history_days=7), now, args.data_dir)
        # Training/download time can consume the entry window. Recheck the real clock.
        now = utc(args.as_of)
        positions = read_positions(args.positions, cfg)
        result = recommendations(bundle, bars, now, positions)
        result["historical_replay"] = bool(args.as_of)
        write_json(args.out / "recommendations.json", result)
        pd.json_normalize(result["recommendations"]).to_csv(args.out / "recommendations.csv", index=False)
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
