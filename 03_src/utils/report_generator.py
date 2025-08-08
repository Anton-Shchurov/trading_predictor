"""
Модуль для автоматического создания отчетов EDA в формате markdown.
Сохраняет все результаты анализа и изображения графиков.
"""

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import contextlib


class EDAReportGenerator:
    """Класс для создания отчетов EDA в формате markdown"""
    
    def __init__(self, project_root: str = None):
        """
        Инициализация генератора отчетов
        
        Args:
            project_root: Путь к корню проекта
        """
        if project_root is None:
            # Автоматически определяем корень проекта
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            
        self.project_root = Path(project_root)
        self.interim_dir = self.project_root / "01_data" / "interim"
        self.images_dir = self.interim_dir / "images" / "raw_data_eda"
        
        # Создаем папки если их нет
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем отчет
        self.report_content = []
        self.figure_counter = 0
        
        # Название файла отчета с временной меткой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_filename = f"eda_report_{timestamp}.md"
        self.report_path = self.interim_dir / self.report_filename
        
    def add_header(self, text: str, level: int = 1):
        """Добавляет заголовок в отчет"""
        header = "#" * level + " " + text
        self.report_content.append(header)
        self.report_content.append("")  # Пустая строка
        
    def add_text(self, text: str):
        """Добавляет текст в отчет"""
        self.report_content.append(text)
        self.report_content.append("")  # Пустая строка
        
    def add_code_block(self, code: str, language: str = "python"):
        """Добавляет блок кода в отчет"""
        self.report_content.append(f"```{language}")
        self.report_content.append(code)
        self.report_content.append("```")
        self.report_content.append("")
        
    def add_table(self, df: pd.DataFrame, caption: str = None):
        """Добавляет таблицу в отчет"""
        if caption:
            self.add_text(f"**{caption}**")
            
        # Конвертируем DataFrame в markdown таблицу
        markdown_table = df.to_markdown(index=True, floatfmt=".4f")
        self.report_content.append(markdown_table)
        self.report_content.append("")
        
    def save_current_figure(self, title: str = None, caption: str = None):
        """
        Сохраняет текущую matplotlib фигуру и добавляет ссылку в отчет
        
        Args:
            title: Заголовок для фигуры
            caption: Подпись к фигуре
        """
        self.figure_counter += 1
        
        # Имя файла изображения
        if title:
            # Убираем специальные символы из названия
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_title = clean_title.replace(' ', '_').lower()
            img_filename = f"fig_{self.figure_counter:02d}_{clean_title}.png"
        else:
            img_filename = f"fig_{self.figure_counter:02d}.png"
            
        img_path = self.images_dir / img_filename
        
        # Сохраняем фигуру
        plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Добавляем ссылку в отчет
        if title:
            self.add_header(title, level=3)
            
        # Относительный путь для markdown
        relative_img_path = f"images/{img_filename}"
        self.report_content.append(f"![{caption or title or f'Figure {self.figure_counter}'}]({relative_img_path})")
        self.report_content.append("")
        
        if caption:
            self.add_text(f"*{caption}*")
            
        return img_path
        
    def capture_output(self, func, *args, **kwargs):
        """
        Захватывает вывод функции и добавляет в отчет
        
        Args:
            func: Функция для выполнения
            *args, **kwargs: Аргументы функции
        """
        # Захватываем stdout
        old_stdout = StringIO()
        with contextlib.redirect_stdout(old_stdout):
            result = func(*args, **kwargs)
        
        output = old_stdout.getvalue()
        if output.strip():
            self.add_code_block(output.strip(), language="")
            
        return result
        
    def add_statistics_summary(self, df: pd.DataFrame, title: str = "Статистическая сводка"):
        """Добавляет статистическую сводку DataFrame"""
        self.add_header(title, level=2)
        
        # Основная статистика
        stats_df = df.describe()
        self.add_table(stats_df, "Описательная статистика")
        
        # Информация о датасете
        info_text = f"""
**Информация о датасете:**
- Размер: {df.shape[0]} строк, {df.shape[1]} столбцов
- Период: с {df.index.min()} по {df.index.max()}
- Пропущенные значения: {df.isnull().sum().sum()}
- Размер в памяти: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
        self.add_text(info_text)
        
    def save_report(self):
        """Сохраняет отчет в файл"""
        # Добавляем заключительную информацию
        self.add_header("Информация о генерации отчета", level=2)
        
        generation_info = f"""
**Дата генерации:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Файл отчета:** {self.report_filename}
**Папка с изображениями:** {self.images_dir.relative_to(self.project_root)}
**Количество изображений:** {self.figure_counter}
        """
        self.add_text(generation_info)
        
        # Записываем файл
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))
            
        print(f"✅ Отчет сохранен: {self.report_path}")
        print(f"📊 Сохранено изображений: {self.figure_counter}")
        print(f"🖼️ Папка с изображениями: {self.images_dir}")
        
        return self.report_path


# Пример использования для notebook
def create_eda_report_context():
    """Создает контекст для генерации отчета EDA"""
    return EDAReportGenerator()


# Утилитарные функции для integration в notebook
def save_figure_to_report(report_gen: EDAReportGenerator, title: str, caption: str = None):
    """Быстрое сохранение текущей фигуры в отчет"""
    return report_gen.save_current_figure(title=title, caption=caption)


def add_analysis_section(report_gen: EDAReportGenerator, title: str, analysis_func, *args, **kwargs):
    """
    Добавляет секцию анализа в отчет
    
    Args:
        report_gen: Генератор отчета
        title: Заголовок секции
        analysis_func: Функция анализа
        *args, **kwargs: Аргументы для функции
    """
    report_gen.add_header(title, level=2)
    
    # Выполняем анализ и захватываем вывод
    result = report_gen.capture_output(analysis_func, *args, **kwargs)
    
    return result 