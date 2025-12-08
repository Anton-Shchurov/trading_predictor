"""
Функции для создания целевой переменной (бинарная цель `y_bs`).

Определения:
- Горизонт прогноза H (в барах)
- r_t = Close_{t+H} - Close_t

Последние H строк остаются без меток (NaN).
"""

from __future__ import annotations

import pandas as pd

__all__ = ["create_binary_labels"]


def create_binary_labels(
    close: pd.Series,
    *,
    horizon: int = 5,
    return_debug: bool = False,
) -> pd.DataFrame:
    """
    Создаёт DataFrame с бинарной целевой переменной `y_bs` (1/0) по знаку приращения Close через H баров.

    1 — buy, если r_t > 0; 0 — sell, если r_t <= 0. Последние H строк — NaN.

    Args:
        close: Серия цен закрытия (DatetimeIndex)
        horizon: Горизонт прогноза H (в барах)
        return_debug: Вернуть ли отладочную колонку f_ret_h

    Returns:
        DataFrame с колонками: y_bs (+ опционально f_ret_h)
    """
    if close is None or len(close) == 0:
        raise ValueError("Серия close пуста")

    close = close.astype(float)

    future_close = close.shift(-horizon)
    ret_h = future_close - close

    valid_mask = ret_h.notna()
    y = pd.Series(pd.NA, index=close.index, dtype="Int8")
    y.loc[valid_mask & (ret_h > 0)] = 1
    y.loc[valid_mask & (ret_h <= 0)] = 0

    out = pd.DataFrame(index=close.index)
    out["y_bs"] = y

    if return_debug:
        out["f_ret_h"] = ret_h

    return out


