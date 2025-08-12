from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class SplitConfig:
    method: str
    ratios: Dict[str, float]
    dates: Dict[str, Optional[str]]
    n_splits: int
    max_train_size: Optional[int]
    test_size: Optional[int]
    gap: int
    artifacts: Dict[str, Optional[str]]


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex в качестве индекса датафрейма")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _validate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Требуем столбцы признаков f_* и целевую переменную y_bhs
    feature_cols = [c for c in df.columns if c.startswith("f_") or c == "close"]
    if "y_bhs" not in df.columns:
        raise ValueError("В датафрейме отсутствует колонка 'y_bhs' — создайте метки целевого класса")

    X = df[feature_cols].astype(float)
    y = df["y_bhs"].astype("Int8")

    # Убираем строки с NaN/inf по X или y
    mask = (~X.isna().any(axis=1)) & (~np.isinf(X).any(axis=1)) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Проверяем, что метки в формате 0,1,2
    unique_labels = set(int(v) for v in pd.Series(y).unique().tolist())
    if not unique_labels.issubset({0, 1, 2}):
        raise ValueError(
            f"Ожидаются метки классов 0,1,2. Обнаружены: {sorted(unique_labels)}. Пересоздайте датасет с новыми метками."
        )
    return X, y


def _temporal_train_valid_test_split(
    X: pd.DataFrame, y: pd.Series, cfg: SplitConfig
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Dict]:
    n = len(X)
    if cfg.method == "ratios":
        n_train = int(n * cfg.ratios.get("train", 0.7))
        n_valid = int(n * cfg.ratios.get("valid", 0.15))
        # Остальное — test
        n_test = n - n_train - n_valid
        if n_test <= 0:
            raise ValueError("Недостаточно данных для теста при указанных долях")

        idx_train_end = X.index[n_train - 1]
        idx_valid_end = X.index[n_train + n_valid - 1]
        idx_test_start = X.index[n_train + n_valid]

        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_valid, y_valid = X.iloc[n_train : n_train + n_valid], y.iloc[n_train : n_train + n_valid]
        X_test, y_test = X.iloc[n_train + n_valid :], y.iloc[n_train + n_valid :]

        split_meta = {
            "method": "ratios",
            "ratios": cfg.ratios,
            "train_end": str(idx_train_end),
            "valid_end": str(idx_valid_end),
            "test_start": str(idx_test_start),
        }
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), split_meta

    elif cfg.method == "dates":
        train_end = cfg.dates.get("train_end")
        valid_end = cfg.dates.get("valid_end")
        test_start = cfg.dates.get("test_start")

        if test_start is None:
            raise ValueError("Для метода 'dates' необходимо указать 'test_start'")

        X_train = X.loc[:train_end] if train_end else X
        y_train = y.loc[:train_end] if train_end else y

        if valid_end:
            X_valid = X.loc[train_end:valid_end]
            y_valid = y.loc[train_end:valid_end]
        else:
            # Если не указан valid_end — делим хвост train на valid
            n_train = int(len(X_train) * 0.85)
            X_valid = X_train.iloc[n_train:]
            y_valid = y_train.iloc[n_train:]
            X_train = X_train.iloc[:n_train]
            y_train = y_train.iloc[:n_train]

        X_test = X.loc[test_start:]
        y_test = y.loc[test_start:]

        split_meta = {
            "method": "dates",
            "train_end": train_end,
            "valid_end": valid_end,
            "test_start": test_start,
        }
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), split_meta

    else:
        raise ValueError("Неизвестный метод разбиения: {cfg.method}")


def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts().to_dict()
    classes = sorted(counts.keys())
    n_total = len(y)
    n_classes = len(classes)
    weights = {}
    for c in classes:
        weights[int(c)] = n_total / (n_classes * counts[c])
    return weights


def _time_series_folds(X: pd.DataFrame, y: pd.Series, cfg: SplitConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(
        n_splits=cfg.n_splits,
        max_train_size=cfg.max_train_size,
        test_size=cfg.test_size,
        gap=cfg.gap,
    )
    folds = list(tscv.split(X))
    return folds


def _save_cv_indices(folds: List[Tuple[np.ndarray, np.ndarray]], path: Path, index: pd.Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for train_idx, valid_idx in folds:
        data.append(
            {
                "train": [str(index[i]) for i in train_idx],
                "valid": [str(index[i]) for i in valid_idx],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    res = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        # log_loss требует корректные вероятности и все классы
        try:
            res["mlogloss"] = log_loss(y_true, y_proba, labels=[0, 1, 2])
        except Exception:
            pass
    return res


def _fit_xgboost(X_tr, y_tr, X_va, y_va, class_weights: Dict[int, float]):
    from xgboost import XGBClassifier
    try:
        from xgboost.callback import EarlyStopping  # noqa: F401
    except Exception:
        EarlyStopping = None  # type: ignore

    # Метки уже в формате 0..2
    sample_weight = y_tr.map(class_weights).astype(float).values
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        learning_rate=0.05,
        n_estimators=2000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    # Совместимость с разными версиями XGBoost
    try:
        if EarlyStopping is not None:
            clf.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=50, save_best=True)],
            )
        else:
            raise TypeError("callbacks not supported")
    except TypeError:
        # Fallback 1: early_stopping_rounds в fit
        try:
            clf.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except TypeError:
            # Fallback 2: попытка передать early_stopping_rounds в конструктор
            try:
                clf = XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    learning_rate=0.05,
                    n_estimators=2000,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=50,  # type: ignore[arg-type]
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
            except TypeError:
                # Fallback 3: без ранней остановки
                clf = XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    learning_rate=0.05,
                    n_estimators=2000,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
    return clf


def _predict_xgboost(clf, X):
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    return pred, proba


def _fit_lightgbm(X_tr, y_tr, X_va, y_va, class_weights: Dict[int, float]):
    import lightgbm as lgb

    # LGBMClassifier умеет class_weight напрямую
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42,
        n_jobs=-1,
    )
    clf.set_params(class_weight=class_weights)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
    return clf


def _fit_catboost(X_tr, y_tr, X_va, y_va, class_weights: Dict[int, float]):
    from catboost import CatBoostClassifier

    # Переводим веса в порядке классов 0,1,2
    class_weights_list = [class_weights[0], class_weights[1], class_weights[2]]

    clf = CatBoostClassifier(
        loss_function="MultiClass",
        learning_rate=0.05,
        iterations=5000,
        depth=6,
        l2_leaf_reg=3.0,
        od_type="Iter",
        od_wait=50,
        verbose=False,
        random_state=42,
        class_weights=class_weights_list,
    )
    clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    return clf


def _predict_catboost(clf, X):
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    return pred, proba


def _fit_and_eval_model(name: str, X_train, y_train, X_valid, y_valid, class_weights: Dict[int, float]):
    if name == "xgboost":
        clf = _fit_xgboost(X_train, y_train, X_valid, y_valid, class_weights)
        pred, proba = _predict_xgboost(clf, X_valid)
        metrics = _evaluate(y_valid.values, pred, proba)
        return clf, metrics

    if name == "lightgbm":
        clf = _fit_lightgbm(X_train, y_train, X_valid, y_valid, class_weights)
        proba = clf.predict_proba(X_valid)
        pred = clf.predict(X_valid)
        metrics = _evaluate(y_valid.values, pred, proba)
        return clf, metrics

    if name == "catboost":
        clf = _fit_catboost(X_train, y_train, X_valid, y_valid, class_weights)
        pred, proba = _predict_catboost(clf, X_valid)
        metrics = _evaluate(y_valid.values, pred, proba)
        return clf, metrics

    raise ValueError(f"Неизвестная модель: {name}")


def _confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True)
    return {"confusion_matrix": cm.tolist(), "classification_report": report}


def run_modeling_pipeline(
    features_path: str = "01_data/processed/eurusd_features.parquet",
    config_splits_path: str = "04_configs/splits.yml",
    save_dir: str = "06_reports",
    models_to_run: Optional[List[str]] = None,
) -> Dict:
    """Полный цикл моделирования и оценки.

    1) Загружаем фичи + целевую переменную
    2) Валидация, сортировка по времени
    3) Разбиение train/valid/test по времени
    4) Генерация фолдов TimeSeriesSplit на train+valid
    5) Балансировка классов через веса
    6) Обучение XGBoost/LightGBM/CatBoost с ранней остановкой
    7) Оценка по метрикам F1_macro, balanced_accuracy, confusion matrix
    8) Сохранение артефактов в отчет
    """

    project_root = Path(__file__).resolve().parents[2]
    features_fp = project_root / features_path
    splits_fp = project_root / config_splits_path
    save_dir_path = project_root / save_dir
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # 1) Загрузка
    if not features_fp.exists():
        raise FileNotFoundError(f"Не найден файл признаков: {features_fp}")
    df = pd.read_parquet(features_fp)
    # Строгая проверка: наличие любой целевой переменной вида y_*
    if not any(col.startswith("y_") for col in df.columns):
        raise ValueError("В датасете отсутствует целевая переменная вида 'y_*'. Добавьте её на этапе feature engineering.")

    # 2) Валидация/сортировка
    df = _ensure_sorted_index(df)
    X, y = _validate_features(df)

    # 3) Загрузка конфигурации сплитов
    cfg_raw = _load_yaml(splits_fp)
    cfg = SplitConfig(
        method=cfg_raw.get("splits", {}).get("method", "ratios"),
        ratios=cfg_raw.get("splits", {}).get("ratios", {"train": 0.7, "valid": 0.15, "test": 0.15}),
        dates=cfg_raw.get("splits", {}).get("dates", {}),
        n_splits=int(cfg_raw.get("time_series_cv", {}).get("n_splits", 5)),
        max_train_size=cfg_raw.get("time_series_cv", {}).get("max_train_size"),
        test_size=cfg_raw.get("time_series_cv", {}).get("test_size"),
        gap=int(cfg_raw.get("time_series_cv", {}).get("gap", 0)),
        artifacts=cfg_raw.get("artifacts", {}),
    )

    # 4) Разбиение
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), split_meta = _temporal_train_valid_test_split(X, y, cfg)

    # 5) Фолды на train (walk-forward)
    X_trval = pd.concat([X_train, X_valid])
    y_trval = pd.concat([y_train, y_valid])
    folds = _time_series_folds(X_trval, y_trval, cfg)
    if cfg.artifacts.get("save_cv_indices", True):
        _save_cv_indices(folds, project_root / cfg.artifacts.get("cv_indices_path", "06_reports/cv_indices.json"), X_trval.index)
    if cfg.artifacts.get("save_split_dates", True):
        with open(project_root / cfg.artifacts.get("split_dates_path", "06_reports/split_dates.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(split_meta, f, allow_unicode=True)

    # 6) Веса классов
    class_weights = _compute_class_weights(y_train)

    # 7) Обучение на каждом фолде и агрегирование метрик (по валидации фолда)
    models = ["xgboost", "lightgbm", "catboost"] if models_to_run is None else models_to_run
    results: Dict[str, Dict] = {}

    for model_name in models:
        fold_metrics: List[Dict[str, float]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
            X_tr, y_tr = X_trval.iloc[tr_idx], y_trval.iloc[tr_idx]
            X_va, y_va = X_trval.iloc[va_idx], y_trval.iloc[va_idx]

            clf, metrics = _fit_and_eval_model(model_name, X_tr, y_tr, X_va, y_va, class_weights)
            fold_metrics.append(metrics)

        # усреднение метрик по фолдам
        avg_metrics = {k: float(np.nanmean([m.get(k, np.nan) for m in fold_metrics])) for k in fold_metrics[0].keys()}

        # финальное дообучение на train+valid и оценка на test
        clf_final, _ = _fit_and_eval_model(model_name, X_trval, y_trval, X_valid, y_valid, class_weights)
        # оценим на тесте
        if model_name == "catboost":
            y_pred_test, y_proba_test = _predict_catboost(clf_final, X_test)
        elif model_name == "xgboost":
            y_pred_test, y_proba_test = _predict_xgboost(clf_final, X_test)
        else:
            y_proba_test = clf_final.predict_proba(X_test)
            y_pred_test = clf_final.predict(X_test)
        test_metrics = _evaluate(y_test.values, y_pred_test, y_proba_test)
        test_details = _confusion_and_report(y_test.values, y_pred_test)

        # сохранение модели
        model_dir = save_dir_path / f"{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_final, model_dir / "model.joblib")

        results[model_name] = {
            "cv_folds": fold_metrics,
            "cv_avg": avg_metrics,
            "test_metrics": test_metrics,
            "test_details": test_details,
        }

    # 8) Сохраняем агрегированный отчет
    with open(save_dir_path / "modeling_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


