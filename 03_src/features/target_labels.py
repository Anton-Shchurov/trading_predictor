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
    atr: pd.Series | None = None,
    *,
    horizon: int = 5,
    k: float = 0.5,
    label_name: str = "y_bs",
    return_debug: bool = False,
) -> pd.DataFrame:
    """
    Создаёт DataFrame с бинарной целевой переменной `y_bs` (1/0) по знаку приращения Close через H баров.
    
    Если передан `atr` (волатильность), то используется динамический барьер:
    y = 1, если (Close[t+H] - Close[t]) / Close[t] >= k * ATR[t] / Close[t]
    
    Args:
        close: Серия цен закрытия (DatetimeIndex)
        atr: Серия ATR (опционально). Если None, используется просто знак приращения.
        horizon: Горизонт прогноза H (в барах)
        k: Коэффициент для ATR порога (используется если atr задан)
        label_name: Имя колонки для целевой переменной (default: "y_bs")
        return_debug: Вернуть ли отладочную колонку f_ret_h
    
    Returns:
        DataFrame с колонками: y_bs (Int8) (+ опционально f_ret_h, threshold)
    """
    if close is None or len(close) == 0:
        raise ValueError("Серия close пуста")

    close = close.astype(float)
    
    # Future Returns: Close[t+H] - Close[t]
    # Используем относительное приращение для корректного сравнения с ATR%
    # ret_h = (Close_{t+H} - Close_t) / Close_t
    future_close = close.shift(-horizon)
    ret_h_abs = future_close - close
    ret_h_rel = ret_h_abs / close.replace(0, np.nan)
    
    valid_mask = ret_h_abs.notna()
    
    # Инициализация Int8 серией с NA
    y = pd.Series(pd.NA, index=close.index, dtype="Int8")
    
    if atr is not None:
        # Логика с ATR барьером
        # threshold = k * ATR_14(t) / Close_t
        # y = 1 if ret_5 >= threshold
        
        atr = atr.astype(float)
        # Выравнивание индексов
        atr = atr.reindex(close.index)
        
        atr_pct = atr / close.replace(0, np.nan)
        threshold = k * atr_pct
        
        # Сравнение
        # 1: Доходность выше порога (положительный выброс)
        # 0: Доходность ниже порога (шум или движение вниз/малое движение вверх)
        # NOTE: В задаче сказано "y_buy_else_atr = 1, если ret_5 >= threshold, y_buy_else_atr = 0 иначе"
        
        mask_pos = (ret_h_rel >= threshold)
        
        y.loc[valid_mask] = 0 # Default to 0 for valid futures
        y.loc[valid_mask & mask_pos] = 1
        
    else:
        # Старая логика (только знак)
        # 1 если return > 0
        y.loc[valid_mask & (ret_h_abs > 0)] = 1
        y.loc[valid_mask & (ret_h_abs <= 0)] = 0

    out = pd.DataFrame(index=close.index)
    out[label_name] = y

    if return_debug:
        out["f_ret_h_rel"] = ret_h_rel.astype('float32')
        if atr is not None:
            out["threshold_atr"] = threshold.astype('float32')

    return out

