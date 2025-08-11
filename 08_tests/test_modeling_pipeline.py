import os
import sys
from pathlib import Path
import pytest

# Добавляем путь к исходникам
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / '03_src'))

from models import prepare_labeled_dataset, run_modeling_pipeline  # noqa: E402


@pytest.mark.slow
def test_prepare_and_modeling_smoke(tmp_path: Path):
    if os.environ.get("TP_RUN_MODELING") != "1":
        pytest.skip("Пропуск тяжёлого smoke-теста моделирования. Установите TP_RUN_MODELING=1 для запуска.")

    # Проверка: файл фич существует
    features_fp = Path(__file__).resolve().parents[2] / "01_data/processed/eurusd_features.parquet"
    assert features_fp.exists(), "Ожидается предварительно сгенерированный файл фич"

    # Подготовка размеченного датасета (если ещё нет)
    labeled_fp = prepare_labeled_dataset(save_path="01_data/processed/eurusd_features_labeled.parquet")
    assert labeled_fp.exists()

    # Запуск пайплайна моделирования (smoke)
    rel_path = str(labeled_fp.relative_to(Path(__file__).resolve().parents[2]))
    results = run_modeling_pipeline(features_path=rel_path, models_to_run=["xgboost"])  # ограничимся одной моделью
    assert isinstance(results, dict)
    assert "xgboost" in results

