"""
Модуль для автоматического логирования результатов экспериментов в experiments.csv.

Собирает данные из YAML конфигов и полученных метрик, записывает в унифицированную таблицу.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _round3(x: Optional[float]) -> Optional[float]:
    """Округление до 3 знаков после запятой с обработкой None и NaN."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return float(np.round(x, 3))


def _infer_asset_tf(fe_cfg: dict) -> str:
    """Извлечение asset/timeframe из конфига feature engineering."""
    try:
        ds = fe_cfg.get("dataset", {})
        active_key = ds.get("active", "")
        item = ds.get("items", {}).get(active_key, {})
        symbol = item.get("symbol")
        tf = item.get("timeframe")
        if symbol and tf:
            return f"{str(symbol).upper()}/{str(tf).upper()}"
    except Exception:
        pass
    # Fallback: парсинг из dataset.active (eurusd_h1_...) -> EURUSD/H1
    try:
        active_key = fe_cfg.get("dataset", {}).get("active", "")
        parts = str(active_key).split("_")
        if len(parts) >= 2:
            return f"{parts[0].upper()}/{parts[1].upper()}"
    except Exception:
        pass
    return "UNKNOWN/TF"


def _infer_ids(fe_cfg: dict) -> Tuple[str, str, str]:
    """Извлечение dataset_id, target_id, fset_id из конфига."""
    dataset_id = str(fe_cfg.get("dataset", {}).get("active", "unknown_dataset"))
    
    # Target ID из feature_definitions
    fd = fe_cfg.get("feature_definitions", {})
    # Поддержка обоих названий целевой переменной
    y_def = fd.get("y_buy_else_atr", fd.get("y_bs", {}))
    params = y_def.get("params", {})
    horizon = params.get("horizon", "?")
    method = y_def.get("method", "binary_target")
    h_part = f"h{int(horizon)}" if isinstance(horizon, (int, float)) else "h?"
    m_part = "binary" if "binary" in str(method).lower() else str(method).lower()
    target_id = f"target_{h_part}_{m_part}"
    
    # Feature set
    fset_id = str(fe_cfg.get("pipeline_settings", {}).get("feature_set", "fset-?"))
    
    return dataset_id, target_id, fset_id


def append_experiment_record(
    experiments_csv_path: Path,
    experiment_id: str,
    results_dict: Dict,
    metrics_dataframe: pd.DataFrame,
    splits_config: dict,
    features_config: dict,
    primary_metric: str = "f1_class_1",
    seed: int = 42,
    model_name: Optional[str] = None,
    params_str: Optional[str] = None,
) -> None:
    """
    Добавляет запись эксперимента в experiments.csv.
    
    Args:
        experiments_csv_path: Путь к файлу experiments.csv
        experiment_id: ID эксперимента (например, "EXP_0020")
        results_dict: Словарь результатов по моделям (без model_obj)
        metrics_dataframe: DataFrame с метриками (колонка "Model" + метрики)
        splits_config: Конфиг разбиения данных (splits.yml)
        features_config: Конфиг фич (feature_engineering.yml)
        primary_metric: Метрика для выбора лучшей модели
        seed: Random seed
        model_name: Явное указание модели (если None - выбирается лучшая)
        params_str: Строка параметров модели для записи
    """
    
    # Определение лучшей модели по primary_metric
    df = metrics_dataframe.copy()
    if primary_metric in df.columns:
        df = df.sort_values(by=primary_metric, ascending=False)
    chosen_model = model_name or (df["Model"].iloc[0] if not df.empty else None)
    
    # Извлечение метрик выбранной модели
    metrics_map: Dict[str, Optional[float]] = {}
    up_pct, down_pct = None, None
    
    if chosen_model and chosen_model in results_dict:
        test_metrics = results_dict[chosen_model].get("test_metrics", {})
        cv_avg = results_dict[chosen_model].get("cv_avg", {})
        
        metrics_map = {
            "accuracy": test_metrics.get("accuracy", cv_avg.get("accuracy")),
            "f1_class_1": test_metrics.get("f1_class_1", cv_avg.get("f1_class_1")),
            "precision_class_1": test_metrics.get("precision_class_1", cv_avg.get("precision_class_1")),
            "recall_class_1": test_metrics.get("recall_class_1", cv_avg.get("recall_class_1")),
            "balanced_accuracy": test_metrics.get("balanced_accuracy", cv_avg.get("balanced_accuracy")),
            "roc_auc": test_metrics.get("roc_auc", cv_avg.get("roc_auc")),
            "simple_pnl": test_metrics.get("simple_pnl", cv_avg.get("simple_pnl")),
            "selected_threshold": test_metrics.get("selected_threshold", cv_avg.get("avg_threshold")),
        }
        
        # Распределение классов из confusion matrix
        details = results_dict[chosen_model].get("test_details", {})
        cm = np.array(details.get("confusion_matrix", []))
        if cm.size > 0:
            supports = cm.sum(axis=1)
            total = float(supports.sum()) if supports.sum() else 1.0
            down_pct = float(supports[0]) / total
            up_pct = float(supports[1]) / total
    
    # Метаданные из конфигов
    asset_tf = _infer_asset_tf(features_config)
    dataset_id, target_id, fset_id = _infer_ids(features_config)
    
    # Схема валидации
    cv_cfg = splits_config.get("time_series_cv", {})
    n_splits = cv_cfg.get("n_splits")
    gap = cv_cfg.get("gap")
    validation_str = f"tscv_k={n_splits}" if n_splits else "tscv"
    if gap:
        validation_str += f"_gap={gap}"
    
    # Колонки таблицы
    columns = [
        "ID", "Date", "Asset/TF", "Dataset", "Target", "Feature set", "Model", "Params",
        "Validation", "Seed", "Acc", "F1", "Precision", "Recall", "BalancedAcc",
        "ROC-AUC", "SimplePnL", "Threshold", "Up %", "Down %", "Primary metric", "Primary value",
    ]
    
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    row = [
        experiment_id,
        now_utc,
        asset_tf,
        dataset_id,
        target_id,
        fset_id,
        chosen_model or "",
        params_str or "",
        validation_str,
        seed,
        _round3(metrics_map.get("accuracy")),
        _round3(metrics_map.get("f1_class_1")),
        _round3(metrics_map.get("precision_class_1")),
        _round3(metrics_map.get("recall_class_1")),
        _round3(metrics_map.get("balanced_accuracy")),
        _round3(metrics_map.get("roc_auc")),
        _round3(metrics_map.get("simple_pnl")),
        _round3(metrics_map.get("selected_threshold")),
        _round3(up_pct),
        _round3(down_pct),
        primary_metric,
        _round3(metrics_map.get(primary_metric)),
    ]
    
    # Запись в CSV
    experiments_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if experiments_csv_path.exists():
        df_csv = pd.read_csv(experiments_csv_path)
        # Добавляем отсутствующие колонки
        for col in columns:
            if col not in df_csv.columns:
                df_csv[col] = None
        df_csv = df_csv.reindex(columns=columns)
        df_csv = pd.concat([df_csv, pd.DataFrame([row], columns=columns)], ignore_index=True)
    else:
        df_csv = pd.DataFrame([row], columns=columns)
    
    df_csv.to_csv(experiments_csv_path, index=False)
    print(f"📝 Experiment record added to: {experiments_csv_path}")
