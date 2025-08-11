from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from features.target_labels import create_bhs_labels


def prepare_labeled_dataset(
    *,
    project_root: Optional[Path] = None,
    features_path: str = "01_data/processed/eurusd_features.parquet",
    ohlc_path: Optional[str] = None,
    config_fe_path: str = "04_configs/feature_engineering.yml",
    horizon: int = 5,
    atr_period: int = 14,
    eps_atr_multiplier: float = 0.2,
    spread: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Path:
    """Готовит размеченный датасет: добавляет колонку y_bhs к фичам и сохраняет.

    Если путь к OHLC не указан, берётся `pipeline_settings.input_file` из
    `04_configs/feature_engineering.yml`.
    """

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    features_fp = project_root / features_path
    if not features_fp.exists():
        raise FileNotFoundError(f"Не найден файл фич: {features_fp}")

    df_features = pd.read_parquet(features_fp)
    if not isinstance(df_features.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex у файла фич")

    # Уже размечен? Просто пересохраняем в целевой путь
    if "y_bhs" in df_features.columns:
        target_fp = project_root / (save_path or features_path.replace(".parquet", "_labeled.parquet"))
        target_fp.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(target_fp)
        return target_fp

    # Определяем путь к OHLC
    if ohlc_path is None:
        cfg_fe_fp = project_root / config_fe_path
        with open(cfg_fe_fp, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ohlc_path = cfg.get("pipeline_settings", {}).get("input_file")
        if not ohlc_path:
            raise ValueError("Не удалось определить путь к OHLC данным из конфигурации")

    ohlc_fp = project_root / ohlc_path
    if not ohlc_fp.exists():
        raise FileNotFoundError(f"Не найден файл OHLC: {ohlc_fp}")

    df_ohlc = pd.read_csv(ohlc_fp)
    # Приводим Time к индексу
    if "Time" in df_ohlc.columns:
        df_ohlc["Time"] = pd.to_datetime(df_ohlc["Time"]) 
        df_ohlc = df_ohlc.set_index("Time").sort_index()

    # Создаём метки по функции
    labels = create_bhs_labels(
        close=df_ohlc["Close"],
        df_ohlc=df_ohlc,
        horizon=horizon,
        atr_period=atr_period,
        eps_atr_multiplier=eps_atr_multiplier,
        spread=spread,
        return_debug=True,
    )

    # Совмещаем по индексу
    df_merged = df_features.join(labels[["y_bhs"]], how="left")
    # Обрезаем хвост без меток (последние H баров)
    df_merged = df_merged[df_merged["y_bhs"].notna()].copy()
    df_merged["y_bhs"] = df_merged["y_bhs"].astype("Int8")

    target_fp = project_root / (save_path or features_path.replace(".parquet", "_labeled.parquet"))
    target_fp.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(target_fp)
    return target_fp


