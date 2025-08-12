import os
import sys
from pathlib import Path
import pytest

# Добавляем путь к исходникам
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / '03_src'))

from models import prepare_labeled_dataset, run_modeling_pipeline  # noqa: E402


@pytest.mark.slow
def test_modeling_smoke_strict_target(tmp_path: Path):
    if os.environ.get("TP_RUN_MODELING") != "1":
        pytest.skip("Пропуск тяжёлого smoke-теста моделирования. Установите TP_RUN_MODELING=1 для запуска.")

    # Проверка: файл фич существует
    features_fp = Path(__file__).resolve().parents[2] / "01_data/processed/eurusd_features.parquet"
    assert features_fp.exists(), "Ожидается предварительно сгенерированный файл фич"

    # Требуем наличие целевой переменной вида y_*
    df = pd.read_parquet(features_fp)
    assert any(c.startswith("y_") for c in df.columns), "В тестовом окружении нужен готовый таргет y_*"

    # Запуск пайплайна моделирования (smoke, одной моделью)
    rel_path = str(features_fp.relative_to(Path(__file__).resolve().parents[2]))
    results = run_modeling_pipeline(features_path=rel_path, models_to_run=["xgboost"])  # ограничимся одной моделью
    assert isinstance(results, dict)
    assert "xgboost" in results

