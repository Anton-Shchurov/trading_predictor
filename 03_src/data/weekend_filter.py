"""Утилита для фильтрации нерабочих часов на форекс.

Запускается из корня проекта TradingPredictor. По умолчанию читает
`01_data/raw/EURUSD_2010-2024_H1_OANDA.csv`, удаляет строки, соответствующие
закрытию рынка OANDA (с пятницы 17:00 до воскресенья 17:00 по Нью-Йорку), и
сохраняет результат в `01_data/processed/EURUSD_2010-2024_H1_no_weekend.parquet`.
Дополнительно строит график по `close` за последние N периодов.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


NY_TZ = "America/New_York"
UTC_TZ = "UTC"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    project_dir = _project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Фильтрация часов, когда рынок OANDA закрыт (пятница 17:00 — воскресенье 17:00,"
            " America/New_York)."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=project_dir / "01_data/raw/EURUSD_2010-2024_H1_OANDA.csv",
        help="Путь к исходному CSV/Parquet с временным столбцом `time`.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_dir / "01_data/processed/EURUSD_2010-2024_H1_no_weekend.parquet",
        help="Путь сохранения очищенного набора в формате Parquet.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="time",
        help="Название столбца с временной меткой.",
    )
    parser.add_argument(
        "--plot-length",
        type=int,
        default=120,
        help="Количество последних периодов для визуализации `close`.",
    )
    parser.add_argument(
        "--close-column",
        default="close",
        help="Столбец, который использовать на графике.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Файл для сохранения графика (PNG). По умолчанию сохраняется рядом с"
            " выходным датасетом с суффиксом `_preview.png`."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Показывать график интерактивно (если поддерживается окружением).",
    )

    return parser.parse_args(argv)


def load_dataset(path: Path, ts_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if ts_column not in df.columns:
        raise KeyError(f"В датасете отсутствует столбец '{ts_column}'.")

    df = df.copy()
    df[ts_column] = pd.to_datetime(df[ts_column], utc=True, errors="raise")
    return df


def filter_oanda_weekend(df: pd.DataFrame, ts_column: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df_local = df.copy()
    df_local[ts_column] = df_local[ts_column].dt.tz_convert(NY_TZ)

    index = df_local[ts_column]
    dow = index.dt.dayofweek
    hour = index.dt.hour
    minute = index.dt.minute

    # Торги закрываются в пятницу в 16:59 (NY), поэтому удаляем всё, что начинается с 17:00.
    is_friday_after_close = (dow == 4) & (hour >= 17)
    is_saturday = dow == 5
    # Воскресенье: рынок открывается в 17:05 (NY). Удаляем более ранние отметки.
    is_sunday_before_open = (dow == 6) & ((hour < 17) | ((hour == 17) & (minute < 5)))

    mask = ~(is_friday_after_close | is_saturday | is_sunday_before_open)
    filtered = df_local.loc[mask].copy()
    filtered[ts_column] = filtered[ts_column].dt.tz_convert(UTC_TZ)
    return filtered


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def plot_tail(
    df: pd.DataFrame,
    ts_column: str,
    value_column: str,
    length: int,
    output_path: Optional[Path],
    show_plot: bool,
) -> None:
    if df.empty:
        print("[WARN] Датасет пуст — график не построен.")
        return

    tail_df = df.sort_values(ts_column).tail(length)
    if tail_df.empty:
        print("[WARN] После сортировки данных не осталось записей для графика.")
        return

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tail_df[ts_column], tail_df[value_column])
    ax.set_xlabel("Время (UTC)")
    ax.set_ylabel(value_column)
    ax.set_title(f"Последние {min(length, len(tail_df))} периодов: {value_column}")
    fig.autofmt_xdate()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] График сохранён в {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    print(f"[INFO] Загрузка данных из {args.input_path}")
    df = load_dataset(args.input_path, args.timestamp_column)
    before_count = len(df)

    df_filtered = filter_oanda_weekend(df, args.timestamp_column)
    after_count = len(df_filtered)
    removed = before_count - after_count
    removed_pct = (removed / before_count * 100) if before_count else 0

    print(
        "[INFO] Удалено строк: "
        f"{removed} ({removed_pct:.2f}% от исходного объёма)."
    )

    print(f"[INFO] Сохранение результата в {args.output_path}")
    save_dataset(df_filtered, args.output_path)

    plot_path = args.plot_output
    if plot_path is None:
        plot_path = args.output_path.with_name(
            f"{args.output_path.stem}_preview.png"
        )

    print(
        f"[INFO] Построение графика {args.close_column} за последние "
        f"{args.plot_length} периодов."
    )
    plot_tail(
        df_filtered,
        args.timestamp_column,
        args.close_column,
        args.plot_length,
        plot_path,
        args.show_plot,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

