# backtest.py
# MA-crossover backtesting engine.
# Usage: python backtest.py [TICKER] [FAST_MA] [SLOW_MA] [TIMEFRAME]
# Examples:
#   python backtest.py EURUSD=X 20 50 1d
#   python backtest.py GC=F 9 21 1h

import sys
import math
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ── Data ────────────────────────────────────────────────────────────────────

def fetch(ticker: str, interval: str = "1d") -> pd.DataFrame:
    period = "2y" if interval == "1d" else "60d"
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ── Strategy ─────────────────────────────────────────────────────────────────

def signals(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """1 = long, -1 = short, 0 = flat."""
    ma_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ma_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    pos = pd.Series(0, index=df.index)
    pos[ma_fast > ma_slow] = 1
    pos[ma_fast < ma_slow] = -1
    # Signal fires on the *next* bar to avoid look-ahead
    return pos.shift(1).fillna(0)


# ── Trade log ────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 long / -1 short
    entry_price: float
    exit_price: float

    @property
    def pnl_pct(self) -> float:
        return self.direction * (self.exit_price - self.entry_price) / self.entry_price

    @property
    def winner(self) -> bool:
        return self.pnl_pct > 0


def build_trades(df: pd.DataFrame, sig: pd.Series) -> List[Trade]:
    trades: List[Trade] = []
    active_dir: int = 0
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0

    for ts, row in df.iterrows():
        s = int(sig.loc[ts])
        price = float(row["Close"])

        if active_dir != 0 and s != active_dir:
            trades.append(Trade(entry_time, ts, active_dir, entry_price, price))
            active_dir = 0

        if active_dir == 0 and s != 0:
            active_dir = s
            entry_time = ts
            entry_price = price

    # Close any open position at last bar
    if active_dir != 0:
        last_ts = df.index[-1]
        trades.append(Trade(entry_time, last_ts, active_dir, entry_price, float(df["Close"].iloc[-1])))

    return trades


# ── Equity curve ─────────────────────────────────────────────────────────────

def equity_curve(trades: List[Trade], starting_capital: float = 10_000.0) -> List[float]:
    equity = [starting_capital]
    for t in trades:
        equity.append(equity[-1] * (1 + t.pnl_pct))
    return equity


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    ticker: str
    interval: str
    fast_ma: int
    slow_ma: int
    n_trades: int
    win_rate: float
    total_return: float
    annualised_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    start: str
    end: str


def _max_drawdown(equity: List[float]) -> float:
    peak = equity[0]
    worst = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > worst:
            worst = dd
    return worst


def _sharpe(trades: List[Trade], periods_per_year: int = 252) -> float:
    rets = [t.pnl_pct for t in trades]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def _profit_factor(trades: List[Trade]) -> float:
    gross_win = sum(t.pnl_pct for t in trades if t.winner)
    gross_loss = abs(sum(t.pnl_pct for t in trades if not t.winner))
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def compute_stats(ticker: str, interval: str, fast: int, slow: int,
                  df: pd.DataFrame, trades: List[Trade]) -> Stats:
    if not trades:
        raise ValueError("No trades generated.")

    eq = equity_curve(trades)
    pnls = [t.pnl_pct for t in trades]
    winners = [t for t in trades if t.winner]
    losers  = [t for t in trades if not t.winner]

    total_ret = (eq[-1] - eq[0]) / eq[0]

    # Annualise: approximate trading days covered
    span_days = (df.index[-1] - df.index[0]).days or 1
    ann_ret = (1 + total_ret) ** (365 / span_days) - 1

    ppyear = {"1d": 252, "1h": 252 * 8, "4h": 252 * 2}.get(interval, 252)

    return Stats(
        ticker=ticker,
        interval=interval,
        fast_ma=fast,
        slow_ma=slow,
        n_trades=len(trades),
        win_rate=len(winners) / len(trades),
        total_return=total_ret,
        annualised_return=ann_ret,
        max_drawdown=_max_drawdown(eq),
        sharpe=_sharpe(trades, ppyear),
        profit_factor=_profit_factor(trades),
        avg_win=sum(t.pnl_pct for t in winners) / len(winners) if winners else 0.0,
        avg_loss=sum(t.pnl_pct for t in losers)  / len(losers)  if losers  else 0.0,
        best_trade=max(pnls),
        worst_trade=min(pnls),
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
    )


# ── Report ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"

def _bar(v: float, width: int = 30) -> str:
    filled = round(abs(v) * width)
    char = "█" if v >= 0 else "░"
    return char * min(filled, width)

def report(s: Stats, trades: List[Trade]) -> str:
    lines = [
        "╔══════════════════════════════════════════════════╗",
        f"║  EMA Crossover Backtest — {s.ticker:<22}║",
        f"║  EMA({s.fast_ma}) × EMA({s.slow_ma})  |  {s.interval}  |  {s.start} → {s.end:<10}║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Trades        {s.n_trades:>6}                               ║",
        f"║  Win rate      {s.win_rate*100:>6.1f}%  {_bar(s.win_rate, 20):<20} ║",
        f"║  Total return  {_pct(s.total_return):>7}                              ║",
        f"║  Ann. return   {_pct(s.annualised_return):>7}                              ║",
        f"║  Max drawdown  {_pct(-s.max_drawdown):>7}                              ║",
        f"║  Sharpe ratio  {s.sharpe:>6.2f}                               ║",
        f"║  Profit factor {s.profit_factor:>6.2f}                               ║",
        f"║  Avg win       {_pct(s.avg_win):>7}   Avg loss  {_pct(s.avg_loss):>7}         ║",
        f"║  Best trade    {_pct(s.best_trade):>7}   Worst     {_pct(s.worst_trade):>7}         ║",
        "╠══════════════════════════════════════════════════╣",
        "║  Last 5 trades                                   ║",
    ]
    for t in trades[-5:]:
        d = "▲ Long " if t.direction == 1 else "▼ Short"
        lines.append(f"║  {d}  {str(t.entry_time.date())!s} → {str(t.exit_time.date())!s}  {_pct(t.pnl_pct):<10}║")
    lines.append("╚══════════════════════════════════════════════════╝")
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────

def run(ticker: str = "EURUSD=X", fast: int = 20, slow: int = 50,
        interval: str = "1d") -> Stats:
    print(f"\nFetching {ticker} [{interval}]…")
    df = fetch(ticker, interval)
    sig = signals(df, fast, slow)
    trades = build_trades(df, sig)
    stats = compute_stats(ticker, interval, fast, slow, df, trades)
    print(report(stats, trades))
    return stats


if __name__ == "__main__":
    args = sys.argv[1:]
    ticker   = args[0] if len(args) > 0 else "EURUSD=X"
    fast_ma  = int(args[1]) if len(args) > 1 else 20
    slow_ma  = int(args[2]) if len(args) > 2 else 50
    interval = args[3] if len(args) > 3 else "1d"
    run(ticker, fast_ma, slow_ma, interval)
