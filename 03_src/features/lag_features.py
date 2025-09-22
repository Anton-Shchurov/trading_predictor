"""
Модуль для создания лаг-фич (lag features).

Создает лаговые признаки на основе исторических значений:
- Лаги цен (Close, Open, High, Low)
- Лаги объемов
- Лаги доходностей
- Разности и отношения лагов
- Скользящие лаги
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class LagFeatures:
    """
    Класс-калькулятор для лаговых признаков.
    """
    def calculate_identity(self, series: pd.Series) -> pd.Series:
        """Возвращает серию без изменений (используется для переименования)."""
        return series