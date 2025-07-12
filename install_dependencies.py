#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для установки всех зависимостей, необходимых для работы с YuE finetuning.
Предназначен для использования в Ubuntu 22.04 с Python 3.11 и CUDA 12.4+.
"""

import os
import sys
import subprocess
import argparse
import platform
import importlib
import pkg_resources
from pathlib import Path


def run_command(cmd, description=None, exit_on_error=True):
    """
    Запускает команду и выводит вывод в реальном времени
    
    Args:
        cmd: команда для выполнения (список или строка)
        description: описание команды для вывода
        exit_on_error: выходить ли из скрипта при ошибке
    """
    if description:
        print(f"\n[УСТАНОВКА] {description}")
    
    if isinstance(cmd, str):
        print(f"Выполнение: {cmd}")
        shell = True
    else:
        print(f"Выполнение: {' '.join(cmd)}")
        shell = False
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=shell
        )
        
        # Вывод результатов в реальном времени
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # Дожидаемся завершения процесса
        exit_code = process.wait()
        
        if exit_code != 0:
            print(f"Ошибка: команда завершилась с кодом {exit_code}")
            if exit_on_error:
                sys.exit(exit_code)
            return False
        return True
    except Exception as e:
        print(f"Ошибка при выполнении команды: {e}")
        if exit_on_error:
            sys.exit(1)
        return False


def check_system():
    """
    Проверяет системные требования
    """
    print("\n[ПРОВЕРКА] Проверка системных требований")
    
    # Проверка ОС
    if platform.system() != "Linux":
        print(f"Предупреждение: Скрипт оптимизирован для Linux, обнаружена ОС {platform.system()}")
    
    # Проверка версии Python
    python_version = platform.python_version()
    print(f"Версия Python: {python_version}")
    if not python_version.startswith("3."):
        print("Ошибка: Требуется Python 3.x")
        sys.exit(1)
    
    major, minor, _ = map(int, python_version.split('.'))
    if major < 3 or (major == 3 and minor < 8):
        print("Предупреждение: Рекомендуется Python 3.8 или выше")
    
    # Проверка наличия pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Ошибка: pip не установлен")
        sys.exit(1)
    
    # Проверка наличия CUDA
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip()
                    print(f"Обнаружен CUDA: {cuda_version}")
                    break
        else:
            print("CUDA не обнаружен. Обучение будет выполняться на CPU (очень медленно).")
    except FileNotFoundError:
        print("nvidia-smi не найден. CUDA не обнаружен или не установлен.")


def install_base_dependencies():
    """
    Устанавливает базовые зависимости Python
    """
    print("\n[УСТАНОВКА] Обновление pip и установка базовых зависимостей")
    
    # Обновление pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
               "Обновление pip")
    
    # Установка базовых пакетов
    base_packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "scikit-learn>=1.2.0"
    ]
    
    run_command([sys.executable, "-m", "pip", "install"] + base_packages, 
               "Установка базовых пакетов")


def install_torch():
    """
    Устанавливает PyTorch с поддержкой CUDA
    """
    print("\n[УСТАНОВКА] PyTorch с поддержкой CUDA")
    
    # Проверка наличия CUDA для выбора правильной версии PyTorch
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        has_cuda = result.returncode == 0
    except FileNotFoundError:
        has_cuda = False
    
    if has_cuda:
        # Установка PyTorch с CUDA
        torch_command = "pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Установка PyTorch без CUDA
        torch_command = "pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu"
    
    run_command(torch_command, "Установка PyTorch")
    
    # Проверка установки PyTorch
    check_torch = """
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
"""
    
    run_command([sys.executable, "-c", check_torch], 
               "Проверка установки PyTorch", 
               exit_on_error=False)


def install_audio_libs():
    """
    Устанавливает библиотеки для работы с аудио
    """
    print("\n[УСТАНОВКА] Библиотеки для работы с аудио")
    
    # Установка системных зависимостей для аудио библиотек (для Ubuntu)
    if platform.system() == "Linux":
        try:
            run_command("apt-get update && apt-get install -y libsndfile1 ffmpeg", 
                       "Установка системных зависимостей для аудио", 
                       exit_on_error=False)
        except:
            print("Предупреждение: Не удалось установить системные зависимости. Возможно, требуются права администратора.")
    
    # Установка Python-библиотек для аудио
    audio_packages = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "audioread>=3.0.0",
        "ffmpeg-python>=0.2.0"
    ]
    
    run_command([sys.executable, "-m", "pip", "install"] + audio_packages, 
               "Установка Python-библиотек для аудио")
    
    # Установка AudioCraft (по желанию)
    print("\nХотите установить AudioCraft от Meta для улучшенного кодирования аудио? (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        run_command("pip install git+https://github.com/facebookresearch/audiocraft.git", 
                   "Установка AudioCraft", 
                   exit_on_error=False)
    
    # Установка EnCodec (по желанию)
    print("\nХотите установить EnCodec от Meta для альтернативного кодирования аудио? (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        run_command("pip install encodec", 
                   "Установка EnCodec", 
                   exit_on_error=False)


def install_transformers():
    """
    Устанавливает библиотеки для работы с трансформерами
    """
    print("\n[УСТАНОВКА] Библиотеки для работы с трансформерами")
    
    transformers_packages = [
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "wandb>=0.15.0"
    ]
    
    run_command([sys.executable, "-m", "pip", "install"] + transformers_packages, 
               "Установка библиотек для трансформеров")
    
    # Установка DeepSpeed
    print("\nУстановка DeepSpeed для распределенного обучения")
    run_command("pip install deepspeed>=0.10.0", 
               "Установка DeepSpeed", 
               exit_on_error=False)


def install_project_requirements():
    """
    Устанавливает зависимости из requirements.txt проекта
    """
    print("\n[УСТАНОВКА] Зависимости из requirements.txt проекта")
    
    # Поиск файла requirements.txt
    req_file = Path("requirements.txt")
    if not req_file.exists():
        req_file = Path("../requirements.txt")
    
    if req_file.exists():
        run_command([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                   "Установка зависимостей из requirements.txt")
    else:
        print("Файл requirements.txt не найден. Пропуск этого шага.")


def compile_helpers():
    """
    Компилирует вспомогательные C++ модули
    """
    print("\n[КОМПИЛЯЦИЯ] Вспомогательные C++ модули")
    
    # Проверка наличия директории с кодом
    core_datasets_dir = Path("core/datasets")
    if core_datasets_dir.exists():
        # Переходим в директорию и компилируем
        current_dir = os.getcwd()
        os.chdir(core_datasets_dir)
        
        if Path("Makefile").exists():
            run_command("make", "Компиляция C++ модулей", exit_on_error=False)
        
        # Возвращаемся в исходную директорию
        os.chdir(current_dir)
    else:
        print("Директория core/datasets не найдена. Пропуск компиляции.")


def check_dependencies():
    """
    Проверяет наличие и совместимость всех необходимых библиотек
    """
    print("\n[ПРОВЕРКА] Проверка установленных зависимостей")
    
    # Список всех необходимых пакетов с минимальными версиями
    required_packages = {
        "numpy": "1.24.0",
        "scipy": "1.10.0",
        "tqdm": "4.65.0",
        "pandas": "2.0.0",
        "matplotlib": "3.7.0",
        "pyyaml": "6.0",
        "requests": "2.28.0",
        "scikit-learn": "1.2.0",
        "torch": "2.0.0",
        "torchvision": "0.15.0",
        "torchaudio": "2.0.0",
        "librosa": "0.10.0",
        "soundfile": "0.12.0",
        "transformers": "4.30.0",
        "tokenizers": "0.13.0",
        "accelerate": "0.20.0",
        "sentencepiece": "0.1.99",
        "peft": "0.4.0",
        "wandb": "0.15.0",
        "deepspeed": "0.10.0"
    }
    
    # Опциональные пакеты
    optional_packages = {
        "audiocraft": None,  # Нет минимальной версии
        "encodec": None,     # Нет минимальной версии
        "bitsandbytes": "0.41.0"
    }
    
    # Проверка установленных пакетов
    print("\nПроверка обязательных пакетов:")
    all_required_installed = True
    
    for package, min_version in required_packages.items():
        try:
            # Пытаемся импортировать пакет
            importlib.import_module(package)
            
            # Проверяем версию
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                print(f"❌ {package}: установлена версия {installed_version}, требуется минимум {min_version}")
                all_required_installed = False
            else:
                print(f"✅ {package}: версия {installed_version} (OK)")
        except (ImportError, pkg_resources.DistributionNotFound):
            print(f"❌ {package}: не установлен")
            all_required_installed = False
    
    # Проверка опциональных пакетов
    print("\nПроверка опциональных пакетов:")
    for package, min_version in optional_packages.items():
        try:
            # Пытаемся импортировать пакет
            importlib.import_module(package)
            
            # Проверяем версию, если указана минимальная
            installed_version = pkg_resources.get_distribution(package).version
            if min_version and pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                print(f"⚠️ {package}: установлена версия {installed_version}, рекомендуется минимум {min_version}")
            else:
                print(f"✅ {package}: версия {installed_version} (OK)")
        except (ImportError, pkg_resources.DistributionNotFound):
            print(f"⚠️ {package}: не установлен (опционально)")
    
    # Проверка совместимости PyTorch и CUDA
    print("\nПроверка совместимости PyTorch и CUDA:")
    try:
        import torch
        print(f"PyTorch версия: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA доступен: {torch.version.cuda}")
            print(f"✅ Количество GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️ CUDA недоступен. Обучение будет выполняться на CPU (очень медленно).")
    except ImportError:
        print("❌ PyTorch не установлен или не может быть импортирован.")
        all_required_installed = False
    
    # Итоговое сообщение
    if all_required_installed:
        print("\n✅ Все обязательные зависимости установлены и совместимы.")
    else:
        print("\n❌ Некоторые обязательные зависимости отсутствуют или несовместимы.")
        print("   Пожалуйста, установите недостающие пакеты или обновите их версии.")


def check_project_structure():
    """
    Проверяет структуру проекта и наличие необходимых файлов и директорий
    """
    print("\n[ПРОВЕРКА] Проверка структуры проекта")
    
    # Основные директории, которые должны существовать
    required_dirs = [
        "core",
        "core/datasets",
        "core/tokenizer",
        "scripts",
        "config",
        "example"
    ]
    
    # Основные файлы, которые должны существовать
    required_files = [
        "scripts/convert.py",
        "scripts/count_tokens.py",
        "scripts/preprocess_data.py",
        "scripts/run_finetune.py",
        "scripts/train_lora.py",
        "config/ds_config_zero2.json",
        "requirements.txt"
    ]
    
    # Проверка директорий
    print("\nПроверка директорий:")
    all_dirs_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ {directory}: найдена")
        else:
            print(f"❌ {directory}: не найдена")
            all_dirs_exist = False
    
    # Проверка файлов
    print("\nПроверка файлов:")
    all_files_exist = True
    for file in required_files:
        if os.path.isfile(file):
            print(f"✅ {file}: найден")
        else:
            print(f"❌ {file}: не найден")
            all_files_exist = False
    
    # Проверка наличия токенизатора
    tokenizer_dirs = ["mm_tokenizer_v0.2_hf"]
    tokenizer_files = ["mm_tokenizer_v0.2_hf/tokenizer.model"]
    
    print("\nПроверка токенизатора:")
    tokenizer_ok = True
    for directory in tokenizer_dirs:
        if os.path.isdir(directory):
            print(f"✅ {directory}: найдена")
        else:
            print(f"❌ {directory}: не найдена")
            tokenizer_ok = False
    
    for file in tokenizer_files:
        if os.path.isfile(file):
            print(f"✅ {file}: найден")
        else:
            print(f"❌ {file}: не найден")
            tokenizer_ok = False
    
    # Итоговое сообщение
    if all_dirs_exist and all_files_exist and tokenizer_ok:
        print("\n✅ Структура проекта корректна.")
    else:
        print("\n⚠️ В структуре проекта отсутствуют некоторые файлы или директории.")
        print("   Это может привести к ошибкам при запуске скриптов.")


def main():
    """
    Основная функция установки всех зависимостей
    """
    parser = argparse.ArgumentParser(
        description="Установка зависимостей для YuE finetuning"
    )
    parser.add_argument("--skip-system-check", action="store_true",
                       help="Пропустить проверку системных требований")
    parser.add_argument("--skip-torch", action="store_true",
                       help="Пропустить установку PyTorch")
    parser.add_argument("--skip-audio", action="store_true",
                       help="Пропустить установку аудио библиотек")
    parser.add_argument("--skip-transformers", action="store_true",
                       help="Пропустить установку библиотек для трансформеров")
    parser.add_argument("--skip-requirements", action="store_true",
                       help="Пропустить установку из requirements.txt")
    parser.add_argument("--skip-compilation", action="store_true",
                       help="Пропустить компиляцию C++ модулей")
    parser.add_argument("--only-check", action="store_true",
                       help="Только проверить зависимости и структуру проекта без установки")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("УСТАНОВКА И ПРОВЕРКА ЗАВИСИМОСТЕЙ ДЛЯ YUE FINETUNING")
    print("=" * 80)
    
    if args.only_check:
        # Только проверка без установки
        check_system()
        check_dependencies()
        check_project_structure()
        print("\n" + "=" * 80)
        print("ПРОВЕРКА ЗАВЕРШЕНА!")
        print("=" * 80)
        return
    
    # Проверка системных требований
    if not args.skip_system_check:
        check_system()
    
    # Установка базовых зависимостей
    install_base_dependencies()
    
    # Установка PyTorch
    if not args.skip_torch:
        install_torch()
    
    # Установка аудио библиотек
    if not args.skip_audio:
        install_audio_libs()
    
    # Установка библиотек для трансформеров
    if not args.skip_transformers:
        install_transformers()
    
    # Установка зависимостей из requirements.txt
    if not args.skip_requirements:
        install_project_requirements()
    
    # Компиляция вспомогательных модулей
    if not args.skip_compilation:
        compile_helpers()
    
    # Финальная проверка всех зависимостей и структуры проекта
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПОСЛЕ УСТАНОВКИ")
    print("=" * 80)
    
    check_dependencies()
    check_project_structure()
    
    print("\n" + "=" * 80)
    print("УСТАНОВКА И ПРОВЕРКА ЗАВЕРШЕНЫ!")
    print("=" * 80)
    print("\nТеперь вы можете запустить скрипты для дообучения модели YuE.")


if __name__ == "__main__":
    main() 