"""Weekly Ichimoku kumo + Chikou breakout scanner (NSE F&O cash names).

Implements the "weekly tuned" configuration:
  dual_fresh bull3 fwd TK ckPx th0.20 vol1.0 HTF SL3.2

A signal fires on the week the full setup (all nine conditions) flips from
false to true. Checked against the 16 signal dates in the 2-year report: 16/16.

Signal is evaluated on the last COMPLETED Friday week; the trade plan is to
fill at the next week's open. Writes data/scan.json for the static web app.

Usage:  python scanner/scan.py            # fetch from Yahoo and write JSON
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))

TENKAN, KIJUN, SENKOU_B, DISP = 9, 26, 52, 26
ATR_N = 14
BULL_MIN_WEEKS = 3
THICK_MIN = 0.20
VOL_N = 20
SL_ATR, SL_CAP_ATR = 3.2, 5.0
RECENT_WEEKS = 8

CONDITIONS = [
    ("price", "Price above cloud", "Weekly close is above the top of the cloud"),
    ("chikou", "Chikou above cloud", "Close is above the cloud as it stood 26 weeks ago"),
    ("bull3", "Cloud bullish 3+ weeks", "Span A above Span B for at least 3 weeks"),
    ("future", "Future cloud bullish", "Undisplaced Span A is above Span B"),
    ("tk", "Tenkan above Kijun", "Trend alignment"),
    ("ckpx", "Chikou above price", "Close is above the close 26 weeks ago"),
    ("thick", "Cloud thickness", "Cloud height is at least 0.20 weekly ATR"),
    ("vol", "Volume", "Week's volume is at least the 20-week average"),
    ("htf", "Monthly gate", "Last completed month: cloud bullish, price and Chikou above it"),
]
KEYS = [c[0] for c in CONDITIONS]

# Monthly version (backtested 2008-2026; values chosen a priori, trail loosened to 4 ATR)
M_BULL_MIN, M_VOL_N, M_MKT_N = 2, 12, 10
M_SL_ATR, M_SL_CAP_ATR = 2.5, 3.5
RECENT_MONTHS = 12
M_CONDITIONS = [
    ("price", "Price above cloud", "Monthly close is above the top of the cloud"),
    ("chikou", "Chikou above cloud", "Close is above the cloud as it stood 26 months ago"),
    ("bull2", "Cloud bullish 2+ months", "Span A above Span B for at least 2 months"),
    ("future", "Future cloud bullish", "Undisplaced Span A is above Span B"),
    ("tk", "Tenkan above Kijun", "Trend alignment"),
    ("ckpx", "Chikou above price", "Close is above the close 26 months ago"),
    ("thick", "Cloud thickness", "Cloud height is at least 0.20 monthly ATR"),
    ("vol", "Volume", "Month's volume is at least the 12-month average"),
    ("weekly", "Weekly agrees", "Last completed week closed above the weekly cloud"),
    ("mkt", "Market gate", "Nifty 500 monthly close is above its 10-month average"),
]
M_KEYS = [c[0] for c in M_CONDITIONS]


def _mid(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    return (high.rolling(n).max() + low.rolling(n).min()) / 2


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Cloud columns are aligned to the bar they are plotted on."""
    out = pd.DataFrame(index=df.index)
    out["tenkan"] = _mid(df.High, df.Low, TENKAN)
    out["kijun"] = _mid(df.High, df.Low, KIJUN)
    out["a_raw"] = (out.tenkan + out.kijun) / 2
    out["b_raw"] = _mid(df.High, df.Low, SENKOU_B)
    out["a"] = out.a_raw.shift(DISP)
    out["b"] = out.b_raw.shift(DISP)
    out["top"] = out[["a", "b"]].max(axis=1, skipna=False)
    out["bot"] = out[["a", "b"]].min(axis=1, skipna=False)
    return out


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def resample(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return daily.resample(rule).agg(agg).dropna(subset=["Close"])


def evaluate(daily: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame | None:
    """Per-week condition table for one stock, using weeks ending on/before asof."""
    daily = daily.dropna(subset=["Close"])
    daily = daily[daily.index <= asof]
    if len(daily) < 60:
        return None
    w = resample(daily, "W-FRI")
    m = resample(daily, "ME")
    ik = ichimoku(w)
    t = pd.DataFrame(index=w.index)
    t["close"] = w.Close
    t["atr"] = atr(w)
    t["top"], t["bot"] = ik.top, ik.bot

    price = w.Close > ik.top
    chikou = w.Close > ik.top.shift(DISP)
    dual = price & chikou
    bull = ik.a > ik.b
    run = bull.groupby((~bull).cumsum()).cumsum()

    t["price"] = price
    t["chikou"] = chikou
    t["bull3"] = run >= BULL_MIN_WEEKS
    t["future"] = ik.a_raw > ik.b_raw
    t["tk"] = ik.tenkan > ik.kijun
    t["ckpx"] = w.Close > w.Close.shift(DISP)
    t["thickness"] = (ik.top - ik.bot) / t.atr
    t["thick"] = t.thickness >= THICK_MIN
    t["vol_ratio"] = w.Volume / w.Volume.rolling(VOL_N).mean()
    t["vol"] = t.vol_ratio >= 1.0
    t["age"] = run
    t["brk"] = (w.Close - ik.top) / t.atr

    # Monthly gate: last month that had fully completed by the week's Friday.
    im = ichimoku(m)
    gate = (im.a > im.b) & (m.Close > im.top) & (m.Close > im.top.shift(DISP))
    known = im.top.notna() & im.top.shift(DISP).notna()
    gate.index = gate.index + pd.Timedelta(days=1)  # usable from the day after month end
    known.index = gate.index
    t["htf"] = gate.astype(float).reindex(t.index, method="ffill").fillna(0).astype(bool)
    t["hist"] = known.astype(float).reindex(t.index, method="ffill").fillna(0).astype(bool)

    # Setup = every condition true. Signal = the setup just flipped on this week.
    t["dual"] = dual
    t["setup"] = t[KEYS].all(axis=1)
    t["signal"] = t.setup & ~t.setup.shift(1, fill_value=False)
    # "First" = first signal since price + Chikou cleared the cloud (this breakout run).
    seen = t.setup.astype(int).groupby((~dual).cumsum()).cumsum()
    t["first"] = t.signal & (seen == 1)
    raw = t.close - SL_ATR * t.atr
    t["stop"] = np.maximum(np.minimum(raw, t.bot), t.close - SL_CAP_ATR * t.atr)
    return t


def evaluate_monthly(daily: pd.DataFrame, mkt_ok: pd.Series, asof: pd.Timestamp) -> pd.DataFrame | None:
    """Per-month condition table for one stock, using months ending on/before asof.

    A signal is the month the full setup turns on for the first time since price
    and Chikou cleared the cloud (one signal per breakout run)."""
    daily = daily.dropna(subset=["Close"])
    daily = daily[daily.index <= asof]
    if len(daily) < 60:
        return None
    m = resample(daily, "ME")
    w = resample(daily, "W-FRI")
    ik = ichimoku(m)
    t = pd.DataFrame(index=m.index)
    t["close"] = m.Close
    t["atr"] = atr(m)
    t["top"], t["bot"] = ik.top, ik.bot

    dual = (m.Close > ik.top) & (m.Close > ik.top.shift(DISP))
    bull = ik.a > ik.b
    run = bull.groupby((~bull).cumsum()).cumsum()
    t["price"] = m.Close > ik.top
    t["chikou"] = m.Close > ik.top.shift(DISP)
    t["bull2"] = run >= M_BULL_MIN
    t["future"] = ik.a_raw > ik.b_raw
    t["tk"] = ik.tenkan > ik.kijun
    t["ckpx"] = m.Close > m.Close.shift(DISP)
    t["thickness"] = (ik.top - ik.bot) / t.atr
    t["thick"] = t.thickness >= THICK_MIN
    t["vol_ratio"] = m.Volume / m.Volume.rolling(M_VOL_N).mean()
    t["vol"] = t.vol_ratio >= 1.0
    t["age"] = run
    t["brk"] = (m.Close - ik.top) / t.atr
    wk = ichimoku(w)  # weeks whose Friday falls on/before the month end
    t["weekly"] = (w.Close > wk.top).astype(float).reindex(m.index, method="ffill").fillna(0).astype(bool)
    t["mkt"] = mkt_ok.astype(float).reindex(m.index, method="ffill").fillna(0).astype(bool)
    t["hist"] = ik.top.notna() & ik.top.shift(DISP).notna() & t.atr.notna()

    t["dual"] = dual
    t["setup"] = t[M_KEYS].all(axis=1) & t["hist"]
    seen = t.setup.astype(int).groupby((~dual).cumsum()).cumsum()
    t["signal"] = t.setup & ~t.setup.shift(1, fill_value=False) & (seen == 1)
    t["first"] = t.signal
    raw = t.close - M_SL_ATR * t.atr
    t["stop"] = np.maximum(np.minimum(raw, t.bot), t.close - M_SL_CAP_ATR * t.atr)
    return t


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repair Yahoo artefacts: one-day moves beyond -45% / +100% that are bad ticks
    (price snaps back within 5 sessions) or unadjusted corporate actions (level shift)."""
    df = df.dropna(subset=["Close"]).copy()
    px = ["Open", "High", "Low", "Close"]
    for _ in range(6):
        r = df.Close.pct_change()
        ev = r[(r < -0.45) | (r > 1.0)]
        if not len(ev):
            break
        i = df.index.get_loc(ev.index[0])
        before = df.Close.iloc[i - 1]
        later = df.Close.iloc[min(i + 5, len(df) - 1)]
        if 0.75 < later / before < 1.33:      # bad tick: drop the glitched sessions
            j = i
            while j < len(df) and not 0.75 < df.Close.iloc[j] / before < 1.33:
                j += 1
            df = df.drop(df.index[i:j])
        else:                                  # level shift: back-adjust earlier history
            k = df.Close.iloc[i] / before
            df.loc[df.index[:i], px] *= k
            df.loc[df.index[:i], "Volume"] /= k
    return df


REAL_CRASHES = {"YESBANK"}  # genuine collapse, not a data artefact


def last_completed_month(now: datetime | None = None) -> pd.Timestamp:
    now = now or datetime.now(IST)
    return pd.Timestamp(now.date()).replace(day=1) - pd.Timedelta(days=1)


def fetch_market() -> pd.Series | None:
    """Monthly Nifty 500 gate: close above its 10-month average."""
    import yfinance as yf

    for _ in range(3):
        try:
            d = yf.download("^CRSLDX", period="max", auto_adjust=False, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            c = d["Close"].dropna()
            if len(c) > 500:
                c.index = pd.to_datetime(c.index).tz_localize(None)
                return c
        except Exception as e:  # noqa: BLE001
            print("index fetch failed:", e, file=sys.stderr)
        time.sleep(5)
    return None


def market_ok(index_close: pd.Series, asof: pd.Timestamp) -> pd.Series:
    mc = index_close[index_close.index <= asof].resample("ME").last().dropna()
    return mc > mc.rolling(M_MKT_N).mean()


def last_completed_friday(now: datetime | None = None) -> pd.Timestamp:
    now = now or datetime.now(IST)
    d = now.date()
    back = (d.weekday() - 4) % 7
    fri = d - timedelta(days=back)
    if back == 0 and now.hour < 16:  # Friday before the close
        fri -= timedelta(days=7)
    return pd.Timestamp(fri)


def load_universe() -> list[tuple[str, str]]:
    with open(ROOT / "scanner" / "universe.csv") as f:
        return [(r["symbol"], r["name"]) for r in csv.DictReader(f)]


def fetch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    todo = list(symbols)
    for attempt in range(3):
        if not todo:
            break
        for i in range(0, len(todo), 40):
            chunk = todo[i : i + 40]
            data = yf.download(
                [s + ".NS" for s in chunk], period="max", interval="1d", auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
            for s in chunk:
                try:
                    df = data[s + ".NS"].dropna(subset=["Close"])
                except KeyError:
                    continue
                if len(df):
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    out[s] = df[["Open", "High", "Low", "Close", "Volume"]]
            time.sleep(1)
        todo = [s for s in symbols if s not in out]
        if todo:
            time.sleep(5 * (attempt + 1))
    return out


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def build(tables: dict[str, pd.DataFrame], names: dict[str, str], asof: pd.Timestamp, tf: str) -> dict:
    """Turn per-stock condition tables into the JSON block for one timeframe."""
    conds_def, keys, n_recent = (CONDITIONS, KEYS, RECENT_WEEKS) if tf == "weekly" else (M_CONDITIONS, M_KEYS, RECENT_MONTHS)
    stocks, recent = [], []
    for sym, t in sorted(tables.items()):
        if t is None or len(t) < 2:
            continue
        row, prev = t.iloc[-1], t.iloc[-2]
        if (asof - t.index[-1]).days > 10:  # stale / suspended
            continue
        conds = {k: bool(row[k]) for k in keys}
        stocks.append({
            "s": sym, "n": names.get(sym, sym),
            "close": r2(row.close), "chg": r2((row.close / prev.close - 1) * 100),
            "c": conds, "pass": int(sum(conds.values())), "signal": bool(row.signal),
            "setup": bool(row.setup), "first": bool(row["first"]), "dual": bool(row.dual), "hist": bool(row["hist"]),
            "thick": r2(row.thickness), "age": int(row.age) if np.isfinite(row.age) else 0,
            "brk": r2(row.brk), "volx": r2(row.vol_ratio), "atr": r2(row.atr),
            "top": r2(row.top), "bot": r2(row.bot), "stop": r2(row.stop),
            "stopPct": r2((row.stop / row.close - 1) * 100),
        })
        hist = t.iloc[-n_recent - 1 : -1]
        for d, h in hist[hist.signal].iterrows():
            after = t.loc[t.index > d]
            below = after.index[after.close < after.bot]
            recent.append({
                "s": sym, "n": names.get(sym, sym), "date": d.strftime("%Y-%m-%d"),
                "first": bool(h["first"]), "sigClose": r2(h.close), "stop": r2(h.stop), "last": r2(row.close),
                "ret": r2((row.close / h.close - 1) * 100), "ago": int(len(after)),
                "status": "Cloud break" if len(below) else ("Stop hit" if (after.close < h.stop).any() else "Open"),
            })
    stocks.sort(key=lambda x: (-x["signal"], -x["setup"], -x["pass"], x["s"]))
    recent.sort(key=lambda x: (x["date"], x["s"]), reverse=True)
    return {
        "periodEnding": asof.strftime("%Y-%m-%d"),
        "universe": len(stocks),
        "conditions": [{"k": k, "label": l, "desc": d} for k, l, d in conds_def],
        "stocks": stocks,
        "recent": recent,
    }


def main() -> int:
    uni = load_universe()
    names = dict(uni)
    prices = fetch([s for s, _ in uni])
    print(f"fetched {len(prices)}/{len(uni)} symbols")
    if len(prices) < len(uni) * 0.6:
        print("too few symbols fetched; keeping previous data", file=sys.stderr)
        return 1
    prices = {s: (df if s in REAL_CRASHES else clean(df)) for s, df in prices.items()}
    path = ROOT / "data" / "scan.json"
    out = {"generated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")}

    wk = last_completed_friday()
    out["weekly"] = build({s: evaluate(df, wk) for s, df in prices.items()}, names, wk, "weekly")

    mo = last_completed_month()
    index_close = fetch_market()
    if index_close is not None:
        ok = market_ok(index_close, mo)
        out["monthly"] = build({s: evaluate_monthly(df, ok, mo) for s, df in prices.items()}, names, mo, "monthly")
        out["monthly"]["marketOk"] = bool(ok.iloc[-1])
    else:  # keep the previous monthly block rather than publish a scan without the market gate
        print("Nifty 500 unavailable; monthly scan not refreshed", file=sys.stderr)
        try:
            out["monthly"] = json.loads(path.read_text())["monthly"]
        except Exception:  # noqa: BLE001
            out["monthly"] = None

    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    for tf in ("weekly", "monthly"):
        b = out[tf]
        if b:
            print(f"{tf} to {b['periodEnding']}: {b['universe']} stocks, signals: {[s['s'] for s in b['stocks'] if s['signal']] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
