"""Модуль моделирования: обучение и валидация моделей.

Содержит пайплайн для разделения по времени, walk-forward CV,
балансировки классов и обучения базовых моделей (XGBoost/LightGBM/CatBoost).
"""

from .modeling_pipeline import run_modeling_pipeline  # noqa: F401
from .dataset_preparation import prepare_labeled_dataset  # noqa: F401


