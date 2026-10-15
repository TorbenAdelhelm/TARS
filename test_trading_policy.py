"""Offline tests for timing, data leakage, portfolio accounting and action gates."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from trading_policy import (
    BASE_FEATURES, Config, backtest, chronological_blocks, clean_bars,
    feature_names, make_features, make_labels, recommendations, train, utc,
)


def synthetic_bars(symbol="SPY", sessions=1, cfg=None):
    cfg = cfg or Config(symbols=(symbol,))
    schedule = mcal.get_calendar("NYSE").schedule("2025-01-02", "2025-05-01").iloc[:sessions]
    rng = np.random.default_rng(17 if symbol == "SPY" else 29)
    frames = []
    last = 100.0
    for _, day in schedule.iterrows():
        idx = pd.date_range(day.market_open, day.market_close, freq=cfg.bar, inclusive="left")
        # Predictable but noisy fixture, exclusively for software testing.
        ret = .0005 * np.sin(np.arange(len(idx)) / 3) + rng.normal(0, .001, len(idx))
        close = last * np.exp(np.cumsum(ret))
        opens = np.r_[last, close[:-1]]
        frames.append(pd.DataFrame({"open": opens, "high": np.maximum(opens, close) * 1.0005,
                                    "low": np.minimum(opens, close) * .9995,
                                    "close": close, "volume": rng.integers(1000, 10000, len(idx))}, index=idx))
        last = close[-1]
    raw = pd.concat(frames)
    return clean_bars(raw, symbol, cfg, raw.index[-1] + 2 * cfg.bar)


class FixedProbabilities:
    def predict_proba(self, x):
        return np.tile([.02, .03, .95], (len(x), 1))


def fixed_bundle(cfg):
    return {"config": vars(cfg), "model": FixedProbabilities(), "temperature": 1.0,
            "features": feature_names(cfg), "class_return_means": [-.002, 0, .002],
            "policy": {"min_favourable_probability": .5, "min_expected_net_bps": 0}}


def test_features_do_not_change_when_the_future_changes():
    cfg = Config(symbols=("SPY",))
    b = synthetic_bars(cfg=cfg)
    before = make_features(b, cfg)
    changed = b.copy()
    changed.loc[changed.index[40]:, ["open", "high", "low", "close"]] *= 1.4
    changed.loc[changed.index[40]:, "volume"] *= 7
    after = make_features(changed, cfg)
    pd.testing.assert_frame_equal(before.iloc[:40][feature_names(cfg)], after.iloc[:40][feature_names(cfg)])
    prefix = make_features(b.iloc[:40], cfg)
    pd.testing.assert_frame_equal(before.iloc[:40][feature_names(cfg)], prefix[feature_names(cfg)])


def test_future_open_labels_and_gaps():
    cfg = Config(symbols=("SPY",))
    b = synthetic_bars(sessions=2, cfg=cfg)
    f = make_labels(make_features(b, cfg), cfg)
    for t, row in f.iterrows():
        assert row.planned_entry_at == t + 2 * cfg.bar
        assert row.planned_exit_at == t + 5 * cfg.bar
        assert row.entry_price == b.loc[row.planned_entry_at, "open"]
        assert row.exit_price == b.loc[row.planned_exit_at, "open"]
        assert row.planned_exit_at < row.session_close
        assert b.loc[row.planned_exit_at, "session"] == row.session
    missing = b.index[30]
    gapped = b.drop(index=missing)
    labels = make_labels(make_features(gapped, cfg), cfg)
    assert not ((labels.index < missing) & (labels.planned_exit_at >= missing)).any()
    features = make_features(gapped, cfg)
    assert features.loc[missing + cfg.bar, BASE_FEATURES].isna().any()


def test_incomplete_and_invalid_bars():
    cfg = Config(symbols=("SPY",))
    b = synthetic_bars(cfg=cfg)
    now = b.index[-1] + cfg.bar / 2
    cleaned = clean_bars(b, "SPY", cfg, now)
    assert b.index[-1] not in cleaned.index
    with pytest.raises(ValueError, match="timezone"):
        clean_bars(b.tz_localize(None), "SPY", cfg, now)
    with pytest.raises(ValueError, match="duplicate"):
        clean_bars(pd.concat([b, b.iloc[[-1]]]), "SPY", cfg, now + cfg.bar)


def test_early_close_calendar_and_dst():
    cfg = Config(symbols=("SPY",))
    # Thanksgiving Friday closes 13:00 New York, including the standard-time offset.
    idx = pd.date_range("2025-11-28T14:30:00Z", "2025-11-28T20:55:00Z", freq="5min")
    raw = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}, index=idx)
    b = clean_bars(raw, "SPY", cfg, utc("2025-11-29T00:00:00Z"))
    assert b.index[-1] == utc("2025-11-28T17:55:00Z")
    assert b.session_close.iloc[-1] == utc("2025-11-28T18:00:00Z")


def test_all_instruments_share_time_blocks_and_labels_are_purged():
    cfg = Config(symbols=("SPY", "QQQ"))
    data = pd.concat([make_labels(make_features(synthetic_bars(s, 30, cfg), cfg), cfg) for s in cfg.symbols])
    blocks, _ = chronological_blocks(data)
    parts = list(blocks.values())
    for a, b in zip(parts[:-1], parts[1:]):
        assert a.planned_exit_at.max() < b.signal_available_at.min()
        assert set(a.session).isdisjoint(b.session)
        assert set(a.symbol) == set(b.symbol) == set(cfg.symbols)


def test_nonoverlapping_trades_and_exact_execution_costs():
    cfg = Config(symbols=("SPY",), max_position_weight=.1, max_gross_weight=.1)
    b = synthetic_bars(cfg=cfg)
    rows = make_labels(make_features(b, cfg), cfg)
    rows["p_favourable"], rows["expected_net_bps"] = .9, 10
    policy = {"min_favourable_probability": .5, "min_expected_net_bps": 0}
    report, trades, curve = backtest(rows, {"SPY": b}, policy, cfg)
    assert len(trades) > 2
    assert (trades.planned_entry_at.iloc[1:].reset_index(drop=True)
            > trades.planned_exit_at.iloc[:-1].reset_index(drop=True)).all()
    expected = trades.exit_price * (1-cfg.cost) / (trades.entry_price * (1+cfg.cost)) - 1
    np.testing.assert_allclose(trades.net_return, expected)
    assert report["total_return"] == pytest.approx(trades.portfolio_pnl.sum())
    assert curve.cash.min() >= -1e-12
    assert (trades.budget <= .101).all()


def test_cash_is_possible_and_has_no_cost():
    cfg = Config(symbols=("SPY",))
    b = synthetic_bars(cfg=cfg)
    rows = make_labels(make_features(b, cfg), cfg)
    rows["p_favourable"], rows["expected_net_bps"] = .2, -10
    report, trades, curve = backtest(rows, {"SPY": b}, {"min_favourable_probability": .5, "min_expected_net_bps": 0}, cfg)
    assert trades.empty and report["total_return"] == 0
    assert {"symbol", "net_return", "planned_exit_at"}.issubset(trades.columns)
    assert curve.equity.eq(1).all()


def test_capacity_cannot_be_borrowed_from_a_future_exit():
    cfg = Config(symbols=("SPY", "QQQ"), cost_bps_per_side=0,
                 max_position_weight=.1, max_gross_weight=.1)
    bars = {s: synthetic_bars(s, cfg=cfg) for s in cfg.symbols}
    # Keep prices constant so mark-to-market drift cannot create spare allocation.
    for b in bars.values():
        b[["open", "high", "low", "close"]] = 100.0
    spy = make_labels(make_features(bars["SPY"], cfg), cfg).iloc[[0]].copy()
    qqq_all = make_labels(make_features(bars["QQQ"], cfg), cfg)
    qqq = qqq_all.iloc[[3, 4]].copy()
    # First QQQ signal is one bar BEFORE SPY's exit. Second is at the exit.
    assert qqq.signal_available_at.iloc[0] < spy.planned_exit_at.iloc[0]
    assert qqq.signal_available_at.iloc[1] == spy.planned_exit_at.iloc[0]
    scored = pd.concat([spy, qqq])
    scored["p_favourable"], scored["expected_net_bps"] = .9, 10
    _, trades, _ = backtest(scored, bars, {"min_favourable_probability": .5, "min_expected_net_bps": 0}, cfg)
    qqq_trade = trades.loc[trades.symbol == "QQQ"]
    assert len(qqq_trade) == 1
    assert qqq_trade.signal_available_at.iloc[0] == spy.planned_exit_at.iloc[0]


def test_live_gates_holdings_and_portfolio_allocation():
    cfg = Config(symbols=("SPY", "QQQ"), max_gross_weight=.1)
    bars = {s: synthetic_bars(s, cfg=cfg).iloc[:31] for s in cfg.symbols}
    now = bars["SPY"].index[-1] + cfg.bar + pd.Timedelta(seconds=10)
    bundle = fixed_bundle(cfg)
    result = recommendations(bundle, bars, now)
    actions = [r["action"] for r in result["recommendations"]]
    assert actions.count("CONSIDER_OPEN_LONG") == 1
    assert sum(r["suggested_additional_weight"] for r in result["recommendations"]) <= .1
    for r in result["recommendations"]:
        assert sum(r["outcome_probabilities"].values()) == pytest.approx(1)
    stale = recommendations(bundle, bars, now + 2 * cfg.bar)
    assert all(r["action"] == "WAIT" for r in stale["recommendations"])
    closed = recommendations(bundle, bars, utc("2025-01-02T23:00:00Z"))
    assert all(r["action"] == "WAIT" for r in closed["recommendations"])
    held = recommendations(bundle, bars, now, {"SPY": {"weight": .08, "exit_at": now + cfg.bar}})
    spy = next(r for r in held["recommendations"] if r["symbol"] == "SPY")
    assert spy["action"] == "HOLD"
    due = recommendations(bundle, bars, now, {"SPY": {"weight": .08, "exit_at": now}})
    assert next(r for r in due["recommendations"] if r["symbol"] == "SPY")["action"] == "EXIT_LONG"
    too_late_model = bundle | {"policy_selected_through": now + cfg.bar}
    with pytest.raises(ValueError, match="future training"):
        recommendations(too_late_model, bars, now)


def test_end_to_end_training_and_saved_model(tmp_path):
    cfg = Config(symbols=("SPY", "QQQ"))
    bars = {s: synthetic_bars(s, 30, cfg) for s in cfg.symbols}
    bundle = train(cfg, bars, {"source": "synthetic software-test fixture"}, tmp_path)
    assert .25 <= bundle["temperature"] <= 8
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "report.json").exists()
    result = recommendations(bundle, bars, bars["SPY"].index[-1] + cfg.bar)
    assert len(result["recommendations"]) == 2
