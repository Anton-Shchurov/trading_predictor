"""
Тестовый скрипт для проверки работы системы автоматической генерации отчетов EDA
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к модулям проекта
current_dir = Path(__file__).parent
src_path = current_dir.parent / "03_src"
sys.path.insert(0, str(src_path))

from utils.report_generator import EDAReportGenerator

def test_report_system():
    """Тестирует систему генерации отчетов"""
    
    print("🧪 Тестирование системы генерации отчетов...")
    
    try:
        # Создаем тестовый отчет
        report = EDAReportGenerator()
        
        # Добавляем контент
        report.add_header('Тестовый отчет системы', level=1)
        report.add_text('Это тестовый отчет для проверки работы системы автоматического сохранения.')
        
        report.add_header('Тестовая секция', level=2)
        report.add_text('Эта секция проверяет добавление текста и заголовков.')
        
        # Создаем тестовую таблицу
        import pandas as pd
        test_data = pd.DataFrame({
            'Параметр': ['Среднее', 'Медиана', 'Стд. отклонение'],
            'Значение': [1.2345, 1.1234, 0.5678]
        })
        report.add_table(test_data, 'Тестовая таблица')
        
        # Создаем тестовые графики
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График 1: Синусоида
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        axes[0, 0].plot(x, y, label='sin(x)')
        axes[0, 0].set_title('Синусоида')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # График 2: Гистограмма
        data = np.random.normal(0, 1, 1000)
        axes[0, 1].hist(data, bins=30, alpha=0.7)
        axes[0, 1].set_title('Нормальное распределение')
        axes[0, 1].set_xlabel('Значение')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].grid(True)
        
        # График 3: Scatter plot
        x_scatter = np.random.randn(100)
        y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
        axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6)
        axes[1, 0].set_title('Корреляция')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].grid(True)
        
        # График 4: Временной ряд
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        values = np.cumsum(np.random.randn(50))
        axes[1, 1].plot(dates, values)
        axes[1, 1].set_title('Временной ряд')
        axes[1, 1].set_xlabel('Дата')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].grid(True)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Сохраняем график в отчет
        report.save_current_figure('Тестовые графики', 'Набор тестовых графиков для проверки работы системы')
        
        plt.close()  # Закрываем фигуру
        
        # Добавляем код
        test_code = """
import pandas as pd
import numpy as np

# Тестовый код
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(data.head())
"""
        report.add_code_block(test_code, 'python')
        
        # Сохраняем отчет
        report_path = report.save_report()
        
        print(f"✅ Тест успешно завершен!")
        print(f"📄 Отчет сохранен: {report_path}")
        print(f"📁 Папка interim: {report.interim_dir}")
        print(f"🖼️ Папка изображений: {report.images_dir}")
        
        # Проверяем созданные файлы
        if report_path.exists():
            print(f"✓ Файл отчета создан: {report_path.stat().st_size} байт")
        
        if report.images_dir.exists():
            image_files = list(report.images_dir.glob("*.png"))
            print(f"✓ Создано изображений: {len(image_files)}")
            for img in image_files:
                print(f"  - {img.name}: {img.stat().st_size} байт")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_report_system()
    
    if success:
        print("\n🎉 Система генерации отчетов работает корректно!")
        print("\n📋 Следующие шаги:")
        print("1. Запустите полный EDA notebook: 02_notebooks/eda/eda.ipynb")
        print("2. Или используйте: python 03_src/utils/quick_report.py")
        print("3. Проверьте созданные файлы в 01_data/interim/")
    else:
        print("\n💥 Тестирование не удалось")
        print("Проверьте установку зависимостей: pandas, matplotlib, pathlib") 