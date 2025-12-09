"""
Функции для создания целевой переменной (бинарная цель `y_bs`).

Определения:
- Горизонт прогноза H (в барах)
- r_t = Close_{t+H} - Close_t

Последние H строк остаются без меток (NaN).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

__all__ = ["create_binary_labels"]


def create_binary_labels(
    close: pd.Series,
    *,
    horizon: int = 5,
    return_debug: bool = False,
) -> pd.DataFrame:
    """
    Создаёт DataFrame с бинарной целевой переменной `y_bs` (1/0) по знаку приращения Close через H баров.

    ВАЖНО: Эта функция создает target для обучения. 
    Она использует БУДУЩИЕ данные (shift(-horizon)).
    Никогда не используйте результат этой функции как входной фич (Feature Leakage).

    Args:
        close: Серия цен закрытия (DatetimeIndex)
        horizon: Горизонт прогноза H (в барах)
        return_debug: Вернуть ли отладочную колонку f_ret_h

    Returns:
        DataFrame с колонками: y_bs (Int8) (+ опционально f_ret_h)
    """
    if close is None or len(close) == 0:
        raise ValueError("Серия close пуста")

    close = close.astype(float)

    # Future Returns: Close[t+H] - Close[t]
    future_close = close.shift(-horizon)
    ret_h = future_close - close

    valid_mask = ret_h.notna()
    
    # Инициализация Int8 серией с NA
    y = pd.Series(pd.NA, index=close.index, dtype="Int8")
    
    # Логика: 1 если return > 0, иначе 0 (включая 0.0)
    # Можно изменить логику на тройную классификацию если нужно (1, 0, -1), но здесь бинарная
    y.loc[valid_mask & (ret_h > 0)] = 1
    y.loc[valid_mask & (ret_h <= 0)] = 0

    out = pd.DataFrame(index=close.index)
    out["y_bs"] = y

    if return_debug:
        out["f_ret_h"] = ret_h.astype('float32')

    return out
