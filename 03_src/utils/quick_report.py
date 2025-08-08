"""
Быстрый запуск генерации отчетов EDA из командной строки.
Использование: python quick_report.py
"""

import sys
from pathlib import Path
import subprocess
import os

def run_eda_notebook():
    """Запускает notebook EDA и генерирует отчет"""
    
    # Определяем пути
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    notebook_path = project_root / "02_notebooks" / "eda" / "eda.ipynb"
    
    print("🚀 Запуск генерации EDA отчета...")
    print(f"📁 Проект: {project_root}")
    print(f"📓 Notebook: {notebook_path}")
    
    if not notebook_path.exists():
        print(f"❌ Notebook не найден: {notebook_path}")
        return False
    
    try:
        # Запускаем notebook через nbconvert
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(notebook_path)
        ]
        
        print("⏳ Выполнение notebook...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print("✅ Notebook успешно выполнен!")
            
            # Проверяем созданные файлы
            interim_dir = project_root / "01_data" / "interim"
            if interim_dir.exists():
                md_files = list(interim_dir.glob("eda_report_*.md"))
                images_dir = interim_dir / "images"
                
                if md_files:
                    latest_report = max(md_files, key=lambda p: p.stat().st_mtime)
                    print(f"📄 Создан отчет: {latest_report}")
                    
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.png"))
                    print(f"🖼️ Сохранено изображений: {len(image_files)}")
                    print(f"📁 Папка с изображениями: {images_dir}")
                    
            return True
            
        else:
            print(f"❌ Ошибка выполнения notebook:")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка запуска: {e}")
        return False
    except FileNotFoundError:
        print("❌ Jupyter не найден. Убедитесь, что Jupyter установлен:")
        print("   pip install jupyter nbconvert")
        return False


def main():
    """Главная функция"""
    print("=" * 60)
    print("📊 ГЕНЕРАТОР EDA ОТЧЕТОВ TRADING PREDICTOR")
    print("=" * 60)
    
    success = run_eda_notebook()
    
    if success:
        print("\n🎉 Генерация отчета завершена успешно!")
        print("\n📋 Что было создано:")
        print("- 📄 Markdown отчет с полным анализом")
        print("- 🖼️ Изображения всех графиков")
        print("- 📁 Все файлы в папке 01_data/interim/")
    else:
        print("\n💥 Ошибка при генерации отчета")
        print("\n🔧 Возможные решения:")
        print("- Проверьте установку Jupyter: pip install jupyter")
        print("- Убедитесь, что все зависимости установлены")
        print("- Проверьте путь к данным в notebook")


if __name__ == "__main__":
    main() 