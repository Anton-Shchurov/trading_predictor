"""
Скрипт для запуска тестов Feature Engineering модулей.

Предоставляет различные режимы запуска тестов:
- Быстрые unit-тесты
- Полные интеграционные тесты
- Тесты производительности
- Тесты с покрытием кода
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_command(cmd, description):
    """Запускает команду и выводит результат."""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} завершено успешно за {execution_time:.2f} сек")
        else:
            print(f"❌ {description} завершено с ошибками (код: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Ошибка выполнения команды: {e}")
        return False


def check_dependencies():
    """Проверяет наличие необходимых зависимостей."""
    print("🔍 Проверка зависимостей...")
    
    required_packages = ['pytest', 'pandas', 'numpy']
    optional_packages = ['pytest-cov', 'pytest-xvs', 'psutil']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package} (ОБЯЗАТЕЛЬНЫЙ)")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️  {package} (опциональный)")
    
    if missing_required:
        print(f"\n❌ Отсутствуют обязательные пакеты: {', '.join(missing_required)}")
        print("Установите их командой:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Отсутствуют опциональные пакеты: {', '.join(missing_optional)}")
        print("Для расширенной функциональности установите их:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True


def run_unit_tests():
    """Запускает быстрые unit-тесты."""
    cmd = "python -m pytest test_feature_engineering.py::TestTechnicalIndicators -v"
    return run_command(cmd, "Unit-тесты технических индикаторов")


def run_statistical_tests():
    """Запускает тесты статистических модулей."""
    cmd = "python -m pytest test_feature_engineering.py::TestStatisticalFeatures -v"
    return run_command(cmd, "Unit-тесты статистических признаков")


def run_lag_tests():
    """Запускает тесты лаг-признаков."""
    cmd = "python -m pytest test_feature_engineering.py::TestLagFeatures -v"
    return run_command(cmd, "Unit-тесты лаг-признаков")


def run_pipeline_tests():
    """Запускает тесты пайплайна."""
    cmd = "python -m pytest test_feature_engineering.py::TestFeatureEngineeringPipeline -v"
    return run_command(cmd, "Unit-тесты пайплайна")


def run_integration_tests():
    """Запускает интеграционные тесты."""
    cmd = "python -m pytest test_integration.py -v"
    return run_command(cmd, "Интеграционные тесты")


def run_all_tests():
    """Запускает все тесты."""
    cmd = "python -m pytest . -v"
    return run_command(cmd, "Все тесты")


def run_quick_tests():
    """Запускает быстрые тесты (исключая медленные)."""
    cmd = "python -m pytest . -v -m 'not slow'"
    return run_command(cmd, "Быстрые тесты")


def run_coverage_tests():
    """Запускает тесты с покрытием кода."""
    cmd = "python -m pytest . --cov=../03_src/features --cov-report=html --cov-report=term-missing"
    return run_command(cmd, "Тесты с покрытием кода")


def run_performance_tests():
    """Запускает тесты производительности."""
    cmd = "python -m pytest test_integration.py::TestFullPipeline::test_pipeline_performance -v -s"
    return run_command(cmd, "Тесты производительности")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Запуск тестов для Feature Engineering модулей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run_tests.py --quick          # Быстрые тесты
  python run_tests.py --unit           # Unit-тесты
  python run_tests.py --integration    # Интеграционные тесты
  python run_tests.py --all            # Все тесты
  python run_tests.py --coverage       # Тесты с покрытием
  python run_tests.py --performance    # Тесты производительности
        """
    )
    
    # Группы тестов
    parser.add_argument('--quick', action='store_true', 
                       help='Запустить быстрые тесты (исключая медленные)')
    parser.add_argument('--unit', action='store_true',
                       help='Запустить только unit-тесты')
    parser.add_argument('--integration', action='store_true',
                       help='Запустить интеграционные тесты')
    parser.add_argument('--all', action='store_true',
                       help='Запустить все тесты')
    parser.add_argument('--coverage', action='store_true',
                       help='Запустить тесты с отчетом о покрытии кода')
    parser.add_argument('--performance', action='store_true',
                       help='Запустить тесты производительности')
    
    # Отдельные модули
    parser.add_argument('--technical', action='store_true',
                       help='Тесты технических индикаторов')
    parser.add_argument('--statistical', action='store_true',
                       help='Тесты статистических признаков') 
    parser.add_argument('--lag', action='store_true',
                       help='Тесты лаг-признаков')
    parser.add_argument('--pipeline', action='store_true',
                       help='Тесты пайплайна')
    
    # Опции
    parser.add_argument('--no-deps-check', action='store_true',
                       help='Пропустить проверку зависимостей')
    
    args = parser.parse_args()
    
    # Проверяем зависимости
    if not args.no_deps_check:
        if not check_dependencies():
            sys.exit(1)
    
    print("\n🧪 ЗАПУСК ТЕСТОВ FEATURE ENGINEERING")
    print("=" * 60)
    
    results = []
    
    # Определяем какие тесты запускать
    if args.quick:
        results.append(run_quick_tests())
    elif args.unit:
        results.extend([
            run_unit_tests(),
            run_statistical_tests(), 
            run_lag_tests(),
            run_pipeline_tests()
        ])
    elif args.integration:
        results.append(run_integration_tests())
    elif args.coverage:
        results.append(run_coverage_tests())
    elif args.performance:
        results.append(run_performance_tests())
    elif args.technical:
        results.append(run_unit_tests())
    elif args.statistical:
        results.append(run_statistical_tests())
    elif args.lag:
        results.append(run_lag_tests())
    elif args.pipeline:
        results.append(run_pipeline_tests())
    elif args.all:
        results.append(run_all_tests())
    else:
        # По умолчанию запускаем быстрые тесты
        print("Режим не указан, запускаем быстрые тесты...")
        results.append(run_quick_tests())
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    failed = total - passed
    
    if failed == 0:
        print(f"✅ Все тесты пройдены успешно! ({passed}/{total})")
        sys.exit(0)
    else:
        print(f"❌ Тесты завершены с ошибками: {failed} неудачных из {total}")
        print(f"✅ Успешно: {passed}")
        print(f"❌ Неудачно: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()