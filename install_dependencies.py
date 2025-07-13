#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для установки всех необходимых зависимостей для проекта YuE fine-tuning.
Скрипт устанавливает только современные версии пакетов, совместимые с
Python 3.11, Ubuntu 22.04 и CUDA 12.6.
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path


def print_status(message):
    """Выводит статусное сообщение в консоль."""
    print(f"[INFO] {message}")


def run_command(command, description=None):
    """Запускает команду и выводит результат."""
    if description:
        print_status(description)
    
    print(f"Выполнение: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        return False
    
    return True


def check_python_version():
    """Проверяет версию Python."""
    print_status("Проверка версии Python")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 11):
        print(f"Ошибка: Требуется Python 3.11+, у вас Python {version_info.major}.{version_info.minor}")
        return False
    
    print(f"Используется Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    return True


def check_cuda():
    """Проверяет наличие CUDA."""
    print_status("Проверка CUDA")
    
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("CUDA недоступна. Установка продолжится, но для обучения потребуется GPU с CUDA.")
            return False
        
        print("CUDA обнаружена.")
        return True
    except Exception as e:
        print(f"Ошибка при проверке CUDA: {e}")
        return False


def install_basic_dependencies():
    """Устанавливает базовые зависимости."""
    print_status("Установка базовых зависимостей")
    
    basic_deps = [
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "pathlib>=1.0.1",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.0",
    ]
    
    return run_command(f"{sys.executable} -m pip install -U {' '.join(basic_deps)}", 
                      "Установка базовых библиотек")


def install_torch():
    """Устанавливает PyTorch с поддержкой CUDA 12.6."""
    print_status("Установка PyTorch")
    
    # Команда для установки PyTorch с поддержкой CUDA 12.6
    torch_cmd = f"{sys.executable} -m pip install --upgrade torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121"
    
    return run_command(torch_cmd, "Установка PyTorch с поддержкой CUDA")


def install_audio_libs():
    """Устанавливает библиотеки для работы с аудио."""
    print_status("Установка библиотек для работы с аудио")
    
    audio_deps = [
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
    ]
    
    success = run_command(f"{sys.executable} -m pip install -U {' '.join(audio_deps)}", 
                         "Установка основных аудио-библиотек")
    
    # Установка дополнительных аудио-кодеков (опционально)
    try:
        print_status("Установка дополнительных аудио-кодеков (EnCodec)")
        encodec_cmd = f"{sys.executable} -m pip install -U encodec"
        run_command(encodec_cmd, "Установка EnCodec")
    except Exception as e:
        print(f"Предупреждение: Не удалось установить EnCodec: {e}")
        print("Это не критично, будет использована упрощённая дискретизация.")
    
    return success


def install_ml_libs():
    """Устанавливает библиотеки для машинного обучения."""
    print_status("Установка библиотек для машинного обучения")
    
    ml_deps = [
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "wandb>=0.16.0",
        "deepspeed>=0.12.0",
    ]
    
    return run_command(f"{sys.executable} -m pip install -U {' '.join(ml_deps)}", 
                      "Установка библиотек машинного обучения")


def install_all_dependencies(args):
    """Устанавливает все необходимые зависимости."""
    if not check_python_version():
        if not args.force:
            print("Установка прервана из-за неподходящей версии Python. Используйте --force для игнорирования.")
            return False
    
    has_cuda = check_cuda()
    if not has_cuda and not args.force:
        print("Предупреждение: CUDA не обнаружена. Продолжение установки для CPU-режима.")
        if not args.cpu_only:
            user_input = input("Продолжить установку для CPU-режима? (y/n): ").lower()
            if user_input != 'y':
                print("Установка прервана.")
                return False
    
    print_status("Начало установки зависимостей")
    
    # Устанавливаем базовые зависимости
    if not install_basic_dependencies():
        print("Ошибка при установке базовых зависимостей")
        return False
    
    # Устанавливаем PyTorch
    if not install_torch():
        print("Ошибка при установке PyTorch")
        return False
    
    # Устанавливаем аудио библиотеки
    if not install_audio_libs():
        print("Ошибка при установке аудио библиотек")
        return False
    
    # Устанавливаем библиотеки для ML
    if not install_ml_libs():
        print("Ошибка при установке библиотек машинного обучения")
        return False
    
    print_status("Все зависимости успешно установлены!")
    return True


def main():
    """Основная функция для запуска установки."""
    parser = argparse.ArgumentParser(description='Установка зависимостей для проекта YuE fine-tuning')
    parser.add_argument('--force', action='store_true', help='Игнорировать проверки и продолжить установку')
    parser.add_argument('--cpu-only', action='store_true', help='Установка только для CPU (без проверки CUDA)')
    args = parser.parse_args()
    
    success = install_all_dependencies(args)
    
    if success:
        print("\n============================================================")
        print("Установка зависимостей завершена успешно!")
        print("Теперь вы можете запускать скрипты из проекта YuE fine-tuning.")
        print("============================================================")
        sys.exit(0)
    else:
        print("\n============================================================")
        print("Установка зависимостей завершилась с ошибками.")
        print("Пожалуйста, проверьте вывод выше для получения информации.")
        print("============================================================")
        sys.exit(1)


if __name__ == "__main__":
    main() 