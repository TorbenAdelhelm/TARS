# Verification of this example

The supplied source passed all 10 offline tests on Windows with Python 3.12.
The tests include an end-to-end model fit using explicitly synthetic fixtures;
those fixtures are not used as market-performance evidence.

Separately, both CLI commands were exercised against Yahoo: `train` downloaded
SPY, QQQ and IWM, trained/calibrated the model, selected the policy and evaluated
the holdout; `recommend` reloaded that saved model and fetched fresh Yahoo bars.
The `examples/` folder contains the chronological evaluation report and a
timestamped recommendation snapshot. Downloaded OHLCV, fitted models and local
caches are generated locally and are excluded from version control.

| Instrument | Downloaded completed bars | Sessions | First bar (UTC) | Last training-download bar (UTC) |
|---|---:|---:|---|---|
| SPY | 3311 | 43 | 2026-08-17 13:30:00+00:00 | 2026-10-15 16:20:00+00:00 |
| QQQ | 3311 | 43 | 2026-08-17 13:30:00+00:00 | 2026-10-15 16:20:00+00:00 |
| IWM | 3311 | 43 | 2026-08-17 13:30:00+00:00 | 2026-10-15 16:20:00+00:00 |

The held-out period contains 7 sessions. Test log loss was
1.022732, compared with 1.039109 for the constant
training-prior baseline. Multiclass Brier score (sum over classes) was
0.613435, compared with 0.624652.
These small-window comparisons are descriptive, not evidence of reliable future calibration.

**Policy selection chose cash.** The validation objective did not support entering
trades under the default execution costs and trade-count requirement. Consequently,
the frozen policy made 0 test trades and returned 0.0%;
the current snapshot returns WAIT for the flat instruments. The threshold above
probability one is an explicit cash-only sentinel, not a claimed probability.

The recommendation snapshot was generated at 2026-10-15 16:30:06.196582+00:00. It is historical
output, not a current recommendation. Run the command again to refresh the clock,
market data, positions and entry-expiry checks. Inspect `examples/report.json` for the exact
dates, costs, assumptions and model limits. The model is for research/paper trading.
