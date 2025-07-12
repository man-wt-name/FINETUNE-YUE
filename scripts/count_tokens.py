#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для подсчета токенов во всех .bin файлах в указанной директории.
Python 3.11+ версия оригинального скрипта count_tokens.sh.
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def get_human_readable_size(path):
    """
    Получает размер файла в человекочитаемом формате (например, 1.2G)
    """
    size_bytes = os.path.getsize(path)
    
    # Конвертируем в читаемый формат
    units = ['B', 'K', 'M', 'G', 'T', 'P']
    size = size_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f}{units[unit_index]}"


def count_tokens_in_file(bin_file, parent_dir, log_dir):
    """
    Запускает подсчет токенов для одного .bin файла
    """
    print(f"Checking mmap file: {bin_file}")
    
    # Получаем размер файла в человекочитаемом формате
    file_size = get_human_readable_size(bin_file)
    print(f"Counting largest mmap file: {bin_file}, size: {file_size}")
    
    # Формируем имя подкаталога для логов, аналогично скрипту bash
    subdir = str(bin_file).replace(parent_dir, '', 1)
    if subdir.startswith('/'):
        subdir = subdir[1:]
    
    # Заменяем / на _ аналогично sed 's/\\//_/g'
    subdir = subdir.replace('/', '_')
    
    # Формируем полный путь к скрипту count_mmap_token.py
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'tools', 'count_mmap_token.py')
    
    # Проверяем существование скрипта
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found")
        return None
    
    # Формируем лог файл
    log_file = os.path.join(log_dir, f"count.{subdir}.log")
    
    # Запускаем скрипт подсчета токенов в фоновом режиме
    cmd = ["python", script_path, "--mmap_path", str(bin_file)]
    print(f"Running: {' '.join(cmd)} > {log_file} 2>&1")
    
    try:
        # Открываем лог файл для записи
        with open(log_file, 'w') as log:
            # Запускаем процесс подсчета токенов
            process = subprocess.Popen(cmd, stdout=log, stderr=log)
            print("Process started in background!")
            return process
    except Exception as e:
        print(f"Error starting process for {bin_file}: {e}")
        return None


def main():
    """
    Основная функция, аналогичная скрипту count_tokens.sh
    """
    parser = argparse.ArgumentParser(description='Count tokens in .bin files')
    parser.add_argument('parent_dir', nargs='?', default='/workspace/dataset/music',
                        help='Parent directory containing .bin files (default: /workspace/dataset/music)')
    
    args = parser.parse_args()
    
    print("Please input parent directory, will count all .bin files...")
    print("Example: python count_tokens.py /workspace/dataset/music")
    
    parent_dir = args.parent_dir
    log_dir = "./count_token_logs/"
    
    # Создаем директорию для логов, если она не существует
    os.makedirs(log_dir, exist_ok=True)
    
    # Ищем все .bin файлы в указанной директории
    bin_files = list(Path(parent_dir).glob("**/*.bin"))
    
    if not bin_files:
        print(f"No .bin files found in {parent_dir}")
        return
    
    # Запускаем обработку файлов с использованием ThreadPoolExecutor
    processes = []
    with ThreadPoolExecutor() as executor:
        for bin_file in bin_files:
            process = count_tokens_in_file(bin_file, parent_dir, log_dir)
            if process:
                processes.append(process)
    
    # Для наглядности выводим общее количество обработанных файлов
    print(f"Started counting tokens in {len(processes)} file(s).")
    print("Check the log files in the directory:", os.path.abspath(log_dir))
    print("All counting processes have been started in the background.")
    
    # Ожидаем завершения всех процессов, если нужно
    # for process in processes:
    #     process.wait()
    # print("All counting processes have completed.")


if __name__ == "__main__":
    main() 