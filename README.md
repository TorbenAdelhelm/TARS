# TARS

Yahoo intraday probability-based trading policy.

A complete Python research example: fetch multiple instruments, train a shared model,
calibrate its probabilities on later data, learn entry thresholds, evaluate on a final
chronological holdout, and write current recommendations. It does not place orders.

The defaults are **US-listed equities/ETFs using NYSE regular trading hours**, five-minute
bars, and a fifteen-minute holding period. SPY, QQQ and IWM are example symbols, not
investment selections. Do not use this calendar unchanged for crypto, FX, futures,
European exchanges or instruments with other trading sessions.

## Install and run

Use Python 3.12, the version used for verification. From this folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe trading_policy.py train --symbols SPY QQQ IWM --out run
```

`requirements-tested.txt` also records the exact dependency versions used in the
supplied verification run. You can install that file instead of `requirements.txt`
to reproduce that environment. [VERIFICATION.md](VERIFICATION.md) describes the real
Yahoo run. The `examples/` folder contains its historical evaluation report and
recommendation snapshot. Train locally to create your own fitted model and data.

This downloads the most recent 59 calendar days of five-minute Yahoo bars. A run fails
clearly if a requested instrument is missing, a class has too few training examples,
or fewer than 28 usable trading sessions are available. It does not substitute invented
prices or silently drop requested instruments.

Get another recommendation using the saved model and fresh Yahoo data:

```powershell
.\.venv\Scripts\python.exe trading_policy.py recommend --model run/model.joblib --out run
```

The program writes JSON to standard output, progress to standard error, and files into
`run`. For a different universe or horizon, train a new model:

```powershell
.\.venv\Scripts\python.exe trading_policy.py train --symbols AAPL MSFT NVDA --holding-bars 2 --cost-bps-per-side 4 --out tech_run
```

Two holding bars means ten minutes at the default five-minute interval. The assumed
per-side cost includes spread, slippage and fees together. It is not estimated from
Yahoo OHLCV. Set it to an independently measured, conservative value for your market,
order size and execution method.

Yahoo's download API documents limited recent intraday history; availability depends
on the interval. One-minute data will generally not supply enough history for this
example's minimum session count. Use an accumulated, timestamped archive for that case.
See the [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

## What is learned

This is a **supervised probability-based decision policy**, not reinforcement learning
or a claim of profitable trading. The program estimates conditional return categories;
it does not randomly sample buy/sell actions. Its recommendation is deterministic given
the learned probabilities, thresholds, holdings and clock.

1. Causal features describe the most recent 1, 3, 6 and 12 bars, volatility, bar range,
   volume relative to its past, distance from session VWAP and time of day. Features
   reset their twelve-bar histories after missing bars and at session boundaries.
2. A pooled gradient-boosted classifier learns probabilities for an unfavourable,
   neutral or favourable **future raw price return**. Instrument indicators allow
   instrument-specific effects while sharing information.
3. Temperature scaling is fitted on a separate, later calibration period. It can
   improve calibration; it cannot guarantee that probabilities remain reliable.
4. Fit-period, within-class return averages approximate the expected return from
   the probabilities. Fit-period 1st/99th percentile clipping limits outliers.
   This is a coarse estimate, not a full distribution or a guaranteed expected profit.
5. A later policy-selection period chooses minimum favourable probability and
   estimated net return. It compares 20 threshold combinations with cash using
   `period_return - 0.5 * maximum_drawdown`, requiring at least eight trades.
6. The final period tests that frozen model, calibration and policy. No parameter
   is selected using that test period. The saved model remains this evaluated model;
   it is not subsequently refitted on the holdout.

The entire instrument universe uses the **same date boundaries**: approximately 50%
fit, 20% calibration, 15% policy selection and 15% test. Labels cannot cross sessions
or block boundaries. Model fitting disables the library's automatic random early-
stopping split. Nothing requires treating dependent time-series rows as independent
Gaussian samples. There is no hash transformation or quantum component.

Calibration follows the separation described in the
[scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

## Exact trading timeline

For a five-minute bar starting at 10:00 New York time:

| Event | Time |
|---|---|
| Bar finishes; its features become available | 10:05 |
| Planned entry, with one full bar of execution delay | 10:10 |
| Planned exit after three holding bars | 10:25 |

Training labels and the backtest use those same future entry and exit opens, with
adverse proportional costs at both fills. No label spans a missing bar or overnight.
The full trade must fit before the session closes; the calendar handles holidays,
daylight-saving changes and scheduled early closes.

If per-side cost is `c`, the simulated net trade return is:

```text
exit_open * (1 - c) / (entry_open * (1 + c)) - 1
```

The favourable class means the raw return exceeds the exact round-trip break-even
hurdle plus a one-basis-point neutral buffer. The unfavourable class falls below the
negative of that hurdle. Everything between those thresholds is neutral. Neutral
does not imply a cost-free trade.

Open prices are execution proxies, not guaranteed fills. The extra bar allows time
to calculate a signal, but Yahoo delivery delays can still make it too late. New-entry
recommendations expire strictly at their stated entry time; the program does not
suggest executing an expired signal immediately.

## Recommendations and holdings

Each row includes probabilities, the estimated net return, the signal's timestamps,
the planned entry and exit, the recommendation and the proposed additional NAV weight.
Possible actions:

| Action | Meaning |
|---|---|
| `CONSIDER_OPEN_LONG` | Entry thresholds and timing checks pass; paper-trading candidate |
| `WAIT` | No qualifying entry, stale data, closed market or insufficient allocation |
| `HOLD` | Recorded position has not reached its fixed exit time |
| `EXIT_LONG` | Recorded position's scheduled holding period has elapsed |
| `REVIEW_RISK` | Current exposure exceeds the entry allocation limits |
| `REVIEW_POSITION` | A recorded holding needs review while the market is closed |

Without a positions file, the program **assumes every instrument is flat**. To evaluate
existing holdings, copy `positions.example.json`, replace its values with actual current
portfolio weights and planned UTC exits, then pass it on each call:

```powershell
.\.venv\Scripts\python.exe trading_policy.py recommend --model run/model.joblib --positions my_positions.json --out run
```

An empty JSON object means flat. This research policy is long-only and assumes the
supplied symbols cover all relevant portfolio exposure. Use actual marked-to-market
weights, not quantities. Positions do not get created just because a recommendation
was generated; an external paper-trading ledger must record fills and pending entries.
The program does not discover broker holdings.

Entries use at most 10% of NAV per instrument and 30% in total by default. If several
instruments qualify, higher estimated net return gets allocation first. Existing
holdings reserve capacity, even when awaiting exit. Allocation must be rechecked at
actual execution. Prices can cause weights to drift above entry limits afterwards.

Do not call the same recommendation repeatedly and interpret it as a new order each
time. A caller should deduplicate by `(symbol, planned_entry_at)`, track pending orders
as reserved exposure, and persist actual fills. There is no broker integration here.

## Outputs and how to assess them

- `recommendations.json` / `.csv`: the current result for each requested instrument.
- `report.json`: assumptions, dates, selected thresholds, probability scores and test results.
- `model.joblib`: reusable fitted model, calibrator and thresholds. Load only your own
  trusted model files; joblib uses a pickle-based format.
- `policy_search.csv`: every threshold candidate and its validation performance.
- `test_predictions.csv`: held-out probabilities and realized future returns.
- `test_trades.csv`: non-overlapping simulated trades including costs.
- `test_equity.csv`: cash, NAV and gross exposure marked at every observed bar open.
- `data/SYMBOL.csv`: the adjusted OHLCV bars used in that run for reproducibility.

The downloader keeps its local metadata/cookie cache in `work/yfinance_cache` relative
to the working directory. That cache is excluded from version control.

Compare test log loss and Brier score with the training-prior baseline; lower is better.
Review the reliability bins to compare predicted and realized favourable frequencies.
The code also compares trading results with cash and a passive initial allocation.
The passive comparison can hold overnight; the intraday policy cannot.

**A cash-only selected policy is a valid result.** Weak probabilities or negative
validation results should not be forced into buy recommendations. Even positive
holdout performance in this short window is not evidence of a durable edge.

The simulation shares cash across instruments, applies costs, prevents simultaneous
positions in the same instrument and sizes using NAV at entry. Its drawdown is based
on bar-open marks and can miss losses within a bar. It does not model limit-order
queues, partial fills, price impact, halts, taxes or broker restrictions. Stop-loss
orders are not included; adding them requires corresponding labels and a new backtest.

Repeatedly changing features after looking at test results makes that period part of
development. Before considering deployment, use a longer data archive, rolling
walk-forward evaluation, cost sensitivity and an independent paper-trading period.
Yahoo is useful for prototyping; this example does not establish a live-data service
guarantee or any data-use rights. Check the provider's terms for your use.

## Offline replay and tests

Saved CSVs need `timestamp,open,high,low,close,volume`, with timezone offsets or `Z` in
every timestamp. Timestamps denote bar starts, not bar ends. All instruments need the
same bar interval and US-equity session convention.

```powershell
.\.venv\Scripts\python.exe trading_policy.py train --symbols SPY QQQ IWM --data-dir run/data --out replay
.\.venv\Scripts\python.exe -m pytest -q
```

For a fixed historical replay clock, append `--as-of 2026-10-15T15:02:00Z`. This filters
out bars that were unfinished then. A saved model cannot be replayed at a time before
its policy was selected. A historical replay does not establish that revised Yahoo
bars were available exactly as stored at that historical time.

Tests use explicitly synthetic fixtures to check feature causality, execution labels,
missing bars, half-days, incomplete bars, shared chronological splits, non-overlapping
trades, cost accounting, cash-only behaviour, stale-data gates and end-to-end training.
They test software correctness, not investment performance.

## Planned extensions

The following are proposed next steps, not features of this implementation:

- Same-time-of-day baselines for each instrument's volume and volatility.
- A past-only filter for ordinary versus unusual market conditions.
- Richer lagged-path features and shorter-history features for the opening session.
- Forecasts for several holding horizons and conditional return magnitudes.
- Repeated chronological walk-forward evaluation on a longer intraday archive.
