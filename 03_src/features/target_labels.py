"""
Функции для создания целевых переменных.

MVP: разметка buy/hold/sell (y_bhs) по правилу:
- Горизонт прогноза H (в барах)
- r_t = Close_{t+H} - Close_t
- ATR(14) по историческим данным (без утечки будущего)
- eps_t = 0.2 * ATR_t + spread_t/2
- Метка: 1 если r_t > eps_t; -1 если r_t < -eps_t; иначе 0

Последние H строк остаются без меток (NaN).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def _compute_atr_wilder(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Вычисляет ATR по Уайлдеру (Wilder) с использованием только прошлых данных.
    Требуются колонки: 'High', 'Low', 'Close'. Индекс — время.
    """
    required = {"High", "Low", "Close"}
    if not required.issubset(df_ohlc.columns):
        missing = required.difference(df_ohlc.columns)
        raise ValueError(f"Для ATR необходимы колонки {required}, отсутствуют: {missing}")

    high = df_ohlc["High"].astype(float)
    low = df_ohlc["Low"].astype(float)
    close = df_ohlc["Close"].astype(float)

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR: EMA c alpha = 1/period (adjust=False, без утечки будущего)
    atr = true_range.ewm(alpha=1.0 / float(period), adjust=False).mean()
    return atr


def _infer_spread_series(
    index: pd.Index,
    df_ohlc: Optional[pd.DataFrame] = None,
    spread: Optional[Union[float, int, pd.Series]] = None,
) -> pd.Series:
    """
    Возвращает серию спрэда, выведенную из:
    - spread (если передан числом или серией)
    - колонки 'Spread'/'spread'/'SPREAD' в df_ohlc
    - иначе нули
    """
    if isinstance(spread, (int, float)):
        return pd.Series(float(spread), index=index)
    if isinstance(spread, pd.Series):
        return spread.reindex(index)

    if df_ohlc is not None:
        for name in ("Spread", "spread", "SPREAD"):
            if name in df_ohlc.columns:
                return df_ohlc[name].astype(float).reindex(index)

    return pd.Series(0.0, index=index)


def create_bhs_labels(
    close: pd.Series,
    *,
    horizon: int = 5,
    atr_series: Optional[pd.Series] = None,
    df_ohlc: Optional[pd.DataFrame] = None,
    atr_period: int = 14,
    eps_atr_multiplier: float = 0.2,
    spread: Optional[Union[float, int, pd.Series]] = None,
    return_debug: bool = True,
) -> pd.DataFrame:
    """
    Создаёт DataFrame с колонкой целевой переменной y_bhs (1/0/-1) по правилу buy/hold/sell.

    Без утечки будущего: ATR и eps_t основаны только на прошлых данных. Последние H строк — NaN.

    Args:
        close: Серия цен закрытия (индекс времени)
        horizon: Горизонт прогноза H (в барах)
        atr_series: Готовая серия ATR; если не указана — будет рассчитана по df_ohlc
        df_ohlc: Оригинальный OHLC датафрейм для расчёта ATR и/или спрэда (опционально)
        atr_period: Период ATR (по умолчанию 14)
        eps_atr_multiplier: Коэффициент при ATR в dead-zone epsilon
        spread: Спрэд (float/series). Если None — будет попытка взять из df_ohlc, иначе 0
        return_debug: Вернуть ли отладочные колонки f_ret_h, f_atr, f_dead_eps

    Returns:
        DataFrame с колонками: y_bhs (+ опционально f_ret_h, f_atр, f_dead_eps)
    """
    if close is None or len(close) == 0:
        raise ValueError("Серия close пуста")

    close = close.astype(float)

    if atr_series is None:
        if df_ohlc is None:
            raise ValueError("Нужно передать либо atr_series, либо df_ohlc для расчёта ATR")
        atr_series = _compute_atr_wilder(df_ohlc, period=atr_period)

    atr_series = atr_series.reindex(close.index)

    # Будущая доходность r_t = C_{t+H} - C_t
    future_close = close.shift(-horizon)
    ret_h = future_close - close

    # Dead zone epsilon: eps_t = k * ATR_t + spread_t/2
    spread_series = _infer_spread_series(close.index, df_ohlc=df_ohlc, spread=spread)
    eps_t = eps_atr_multiplier * atr_series + (spread_series / 2.0)

    # Без утечек: последние H строк остаются NaN
    valid_mask = ret_h.notna() & eps_t.notna()

    y = pd.Series(pd.NA, index=close.index, dtype="Int8")
    y.loc[valid_mask & (ret_h > eps_t)] = 1
    y.loc[valid_mask & (ret_h < -eps_t)] = -1
    y.loc[valid_mask & ~(ret_h > eps_t) & ~(ret_h < -eps_t)] = 0

    out = pd.DataFrame(index=close.index)
    out["y_bhs"] = y

    if return_debug:
        out["f_ret_h"] = ret_h
        out["f_atr"] = atr_series
        out["f_dead_eps"] = eps_t

    return out


