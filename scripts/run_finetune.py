#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python 3.11+ версия скрипта run_finetune.sh для запуска процесса дообучения модели YuE.
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import re
from pathlib import Path


def print_help():
    """
    Выводит справочную информацию о скрипте
    """
    help_text = """
========================================================
YuE Fine-tuning Script Help
========================================================
Before running this script, please update the following configuration in the script:

1. Data paths:
   DATA_PATH - Replace <weight_and_path_to_data_X> with actual weights and data paths
   DATA_CACHE_PATH - Replace <path_to_data_cache> with actual cache directory

2. Model configuration:
   TOKENIZER_MODEL_PATH - Replace <path_to_tokenizer_model> with actual tokenizer path
   MODEL_CACHE_DIR - Replace <path_to_model_cache> with actual cache directory
   OUTPUT_DIR - Replace <path_to_output_dir> with actual output directory

3. If using WandB:
   WANDB_API_KEY - Replace <your_wandb_api_key> with your actual API key

Example configuration:
  DATA_PATH = "data1-weight /path/to/data1 data2-weight /path/to/data2"
  DATA_CACHE_PATH = "/path/to/cache"
  TOKENIZER_MODEL_PATH = "/path/to/tokenizer"
  MODEL_CACHE_DIR = "/path/to/model/cache"
  OUTPUT_DIR = "/path/to/output"
  WANDB_API_KEY = "your-actual-wandb-key"
========================================================
"""
    print(help_text)
    sys.exit(1)


def check_placeholders(config):
    """
    Проверяет наличие заполнителей в конфигурации
    """
    has_placeholders = False
    
    if "<weight_and_path_to_data" in config["DATA_PATH"]:
        print("Error: Please set actual weight and data paths in DATA_PATH variable.")
        has_placeholders = True
    
    if "<path_to_data_cache>" in config["DATA_CACHE_PATH"]:
        print("Error: Please set actual data cache path in DATA_CACHE_PATH variable.")
        has_placeholders = True
    
    if "<path_to_tokenizer_model>" in config["TOKENIZER_MODEL_PATH"]:
        print("Error: Please set actual tokenizer model path in TOKENIZER_MODEL_PATH variable.")
        has_placeholders = True
    
    if "<path_to_model_cache>" in config["MODEL_CACHE_DIR"]:
        print("Error: Please set actual model cache directory in MODEL_CACHE_DIR variable.")
        has_placeholders = True
    
    if "<path_to_output_dir>" in config["OUTPUT_DIR"]:
        print("Error: Please set actual output directory in OUTPUT_DIR variable.")
        has_placeholders = True
    
    if config["USE_WANDB"] and "<your_wandb_api_key>" in config["WANDB_API_KEY"]:
        print("Error: Please set actual WandB API key in WANDB_API_KEY variable or disable WandB.")
        has_placeholders = True
    
    if has_placeholders:
        print("\nPlease update the script with your actual paths and values.")
        print("Run 'python scripts/run_finetune.py --help' for more information.")
        sys.exit(1)


def check_environment():
    """
    Проверяет окружение и наличие необходимых зависимостей
    """
    # Проверка, что мы находимся в правильной директории
    current_dir = os.getcwd().lower()
    expected_dirs = ["finetuning-yue", "kaggle/working/finetuning-yue"]
    
    if not any(dir_name in current_dir for dir_name in expected_dirs):
        print(f"Warning: This script is expected to run from the finetuning-yue directory.")
        print(f"Current directory: {os.getcwd()}")
        print("Continuing anyway, but be aware of potential path issues.")
    
    # Проверка наличия torchrun
    try:
        result = subprocess.run(
            ["which", "torchrun"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode != 0:
            print("Error: torchrun not found. Please install PyTorch with distributed support.")
            print("Try: pip install torch>=2.0.0")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking for torchrun: {e}")
        print("Please make sure PyTorch with distributed support is installed.")
    
    # Проверка наличия CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Training will be extremely slow on CPU.")
        else:
            print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("Warning: Could not import torch to check CUDA availability.")
    
    # Проверка наличия DeepSpeed
    try:
        import deepspeed
        print(f"DeepSpeed version {deepspeed.__version__} found.")
    except ImportError:
        print("Warning: DeepSpeed not found. It's required for efficient training.")
        print("Try: pip install deepspeed")


def run_command(cmd):
    """
    Запускает команду и выводит вывод в реальном времени
    """
    print(f"Running command: {cmd}")
    
    try:
        # Запуск процесса с выводом в реальном времени
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=True  # Используем shell=True для сложных команд с перенаправлениями
        )
        
        # Вывод результатов в реальном времени
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # Дожидаемся завершения процесса
        exit_code = process.wait()
        return exit_code
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1


def main():
    """
    Основная функция для запуска дообучения модели YuE
    """
    # Создаем собственный парсер аргументов, не конфликтующий со встроенными опциями
    if len(sys.argv) > 1 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print_help()
        sys.exit(0)
    
    # Проверка окружения и зависимостей
    check_environment()
    
    # ==============================
    # Configuration Parameters
    # ==============================
    
    # Конфигурация оборудования
    config = {
        "NUM_GPUS": 8,
        "MASTER_PORT": 9999,
        # CUDA_VISIBLE_DEVICES можно установить при необходимости
        
        # Гиперпараметры обучения
        "PER_DEVICE_TRAIN_BATCH_SIZE": 1,
        "PER_DEVICE_EVAL_BATCH_SIZE": 1,
        "USE_BF16": True,
        "SEQ_LENGTH": 8192,
        "TRAIN_ITERS": 150,
        "NUM_TRAIN_EPOCHS": 10,
        
        # Пути к данным (замените на свои)
        "DATA_PATH": "<weight_and_path_to_data_X>",
        "DATA_CACHE_PATH": "<path_to_data_cache>",
        
        # Пропорции разделения данных
        "DATA_SPLIT": "900,50,50",
        
        # Конфигурация модели
        "TOKENIZER_MODEL_PATH": "<path_to_tokenizer_model>",
        "MODEL_NAME": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "MODEL_CACHE_DIR": "<path_to_model_cache>",
        "OUTPUT_DIR": "<path_to_output_dir>",
        "DEEPSPEED_CONFIG": "config/ds_config_zero2.json",
        
        # Конфигурация LoRA
        "LORA_R": 64,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.1,
        "LORA_TARGET_MODULES": "q_proj k_proj v_proj o_proj",
        
        # Конфигурация логирования
        "LOGGING_STEPS": 5,
        "SAVE_STEPS": 5,
        "USE_WANDB": True,
        "WANDB_API_KEY": "<your_wandb_api_key>",
        "RUN_NAME": "YuE-ft-lora"
    }
    
    # Глобальный размер батча
    config["GLOBAL_BATCH_SIZE"] = config["NUM_GPUS"] * config["PER_DEVICE_TRAIN_BATCH_SIZE"]
    
    # Проверка заполнителей
    check_placeholders(config)
    
    # ==============================
    # Environment Setup
    # ==============================
    
    # Установка переменных окружения
    os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]
    os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
    
    # Вывод конфигурации
    print("===============================================")
    print("YuE Fine-tuning Configuration:")
    print("===============================================")
    print(f"Number of GPUs: {config['NUM_GPUS']}")
    print(f"Global batch size: {config['GLOBAL_BATCH_SIZE']}")
    print(f"Model: {config['MODEL_NAME']}")
    print(f"Output directory: {config['OUTPUT_DIR']}")
    print(f"Training epochs: {config['NUM_TRAIN_EPOCHS']}")
    print("===============================================")
    
    # ==============================
    # Build and Execute Command
    # ==============================
    
    # Базовая команда
    cmd = f"""torchrun --nproc_per_node={config['NUM_GPUS']} --master_port={config['MASTER_PORT']} scripts/train_lora.py \
        --seq-length {config['SEQ_LENGTH']} \
        --data-path "{config['DATA_PATH']}" \
        --data-cache-path {config['DATA_CACHE_PATH']} \
        --split {config['DATA_SPLIT']} \
        --tokenizer-model {config['TOKENIZER_MODEL_PATH']} \
        --global-batch-size {config['GLOBAL_BATCH_SIZE']} \
        --per-device-train-batch-size {config['PER_DEVICE_TRAIN_BATCH_SIZE']} \
        --per-device-eval-batch-size {config['PER_DEVICE_EVAL_BATCH_SIZE']} \
        --train-iters {config['TRAIN_ITERS']} \
        --num-train-epochs {config['NUM_TRAIN_EPOCHS']} \
        --logging-steps {config['LOGGING_STEPS']} \
        --save-steps {config['SAVE_STEPS']} \
        --deepspeed {config['DEEPSPEED_CONFIG']}"""
    
    # Добавляем условные аргументы
    if config["USE_WANDB"]:
        cmd += f" --report-to wandb --run-name \"{config['RUN_NAME']}\""
    else:
        cmd += " --report-to none"
    
    cmd += f""" \
        --model-name-or-path "{config['MODEL_NAME']}" \
        --cache-dir {config['MODEL_CACHE_DIR']} \
        --output-dir {config['OUTPUT_DIR']} \
        --lora-r {config['LORA_R']} \
        --lora-alpha {config['LORA_ALPHA']} \
        --lora-dropout {config['LORA_DROPOUT']} \
        --lora-target-modules {config['LORA_TARGET_MODULES']}"""
    
    if config["USE_BF16"]:
        cmd += " --bf16"
    
    # Выполнение команды
    print(f"Running command: {cmd}")
    print("===============================================")
    
    # Запускаем команду
    exit_code = run_command(cmd)
    
    # Проверяем результат выполнения
    if exit_code == 0:
        print("===============================================")
        print("Fine-tuning completed successfully!")
        print(f"Output saved to: {config['OUTPUT_DIR']}")
        print("===============================================")
    else:
        print("===============================================")
        print(f"Error: Fine-tuning failed with exit code {exit_code}")
        print("===============================================")
        sys.exit(1)


if __name__ == "__main__":
    main() 