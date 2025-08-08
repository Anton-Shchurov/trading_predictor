"""
Feature Engineering модули для TradingPredictor.

Этот пакет содержит классы для создания различных типов фич:
- TechnicalIndicators: технические индикаторы (EMA, MACD, RSI и др.)
- StatisticalFeatures: статистические фичи (ROC, Z-Score, Skewness и др.)
- LagFeatures: лаг-фичи для временных рядов
"""

from .technical_indicators import TechnicalIndicators
from .statistical_features import StatisticalFeatures
from .lag_features import LagFeatures
from .feature_pipeline import FeatureEngineeringPipeline

__all__ = [
    'TechnicalIndicators',
    'StatisticalFeatures', 
    'LagFeatures',
    'FeatureEngineeringPipeline'
]