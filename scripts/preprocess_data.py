#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python 3.11+ версия скрипта preprocess_data.sh для предобработки данных для YuE.
Преобразует данные в формат, подходящий для обучения модели.
"""

import os
import sys
import argparse
import subprocess
import time
import glob
from pathlib import Path


def run_command(cmd):
    """
    Запускает команду и выводит вывод в реальном времени
    """
    print(f"Running command: {' '.join(cmd)}")
    # Небольшая задержка перед запуском, аналогично sleep 5 в bash
    time.sleep(5)
    
    try:
        # Запуск процесса с выводом в реальном времени
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=False
        )
        
        # Вывод результатов в реальном времени
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # Дожидаемся завершения процесса
        exit_code = process.wait()
        
        if exit_code != 0:
            print(f"Command failed with exit code {exit_code}")
            sys.exit(exit_code)
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def remove_files(pattern):
    """
    Удаляет файлы по шаблону, используя glob для кросс-платформенности
    """
    try:
        # Используем glob для получения списка файлов по шаблону
        files = glob.glob(pattern)
        if files:
            for file in files:
                try:
                    os.remove(file)
                    print(f"Removed file: {file}")
                except OSError as e:
                    print(f"Warning: Failed to remove file {file}: {e}")
        else:
            print(f"No files found matching pattern: {pattern}")
    except Exception as e:
        print(f"Warning: Error while removing files matching {pattern}: {e}")


def preprocess_cot_mode(data_root, name_prefix, tokenizer_model, codec_type, 
                      instruction, order, dropout, quantizer_begin_idx, 
                      num_quantizers, keep_sequential_samples):
    """
    Обрабатывает данные в режиме Chain-of-Thought (CoT)
    """
    print("Running in 'cot' mode...")
    name_suffix = "stage_1_token_level_interleave_cot_xcodec"
    mmap_name = f"mmap/{name_prefix}_{name_suffix}_{order}"
    
    # Удаляем старые файлы и создаем директорию
    remove_files(f"{data_root}/jsonl/{name_prefix}_*.jsonl")
    os.makedirs(f"{data_root}/{mmap_name}", exist_ok=True)
    
    # Строим команду
    cmd = [
        "python", "core/preprocess_data_conditional_xcodec_segment.py",
        "--input", f"{data_root}/jsonl/{name_prefix}.jsonl",
        "--output-prefix", f"{data_root}/{mmap_name}",
        "--tokenizer-model", tokenizer_model,
        "--tokenizer-type", "MMSentencePieceTokenizer",
        "--codec-type", codec_type,
        "--workers", "8",
        "--partitions", "1",
        "--instruction", instruction,
        "--instruction-dropout-rate", str(dropout),
        "--order", order,
        "--append-eod",
        "--quantizer-begin", str(quantizer_begin_idx),
        "--n-quantizer", str(num_quantizers),
        "--use-token-level-interleave",
        "--keep-sequential-samples",
        "--cot"
    ]
    
    # Запускаем команду
    run_command(cmd)
    
    # Удаляем временные файлы
    remove_files(f"{data_root}/jsonl/{name_prefix}_*.jsonl")
    remove_files(f"{data_root}/{mmap_name}_*_text_document.bin")
    remove_files(f"{data_root}/{mmap_name}_*_text_document.idx")


def preprocess_icl_cot_mode(data_root, name_prefix, tokenizer_model, codec_type, 
                          instruction, order, dropout, quantizer_begin_idx, 
                          num_quantizers, keep_sequential_samples, audio_prompt_modes):
    """
    Обрабатывает данные в режиме In-Context Learning Chain-of-Thought (ICL CoT)
    """
    print("Running in 'icl_cot' mode...")
    name_suffix = "stage_1_token_level_interleave_long_prompt_msa"
    mmap_name = f"mmap/{name_prefix}_{name_suffix}_{order}"
    prompt_len = 30
    
    # Удаляем старые файлы и создаем базовую директорию
    remove_files(f"{data_root}/jsonl/{name_prefix}_*.jsonl")
    os.makedirs(f"{data_root}/{mmap_name}", exist_ok=True)
    
    # Обрабатываем каждый режим аудио промптов
    for mode in audio_prompt_modes:
        print(f"Processing mode: {mode}")
        mode_mmap_name = f"{mmap_name}_{mode}"
        os.makedirs(f"{data_root}/{mode_mmap_name}", exist_ok=True)
        
        # Строим команду для этого режима
        cmd = [
            "python", "core/preprocess_data_conditional_xcodec_segment.py",
            "--input", f"{data_root}/jsonl/{name_prefix}.jsonl",
            "--output-prefix", f"{data_root}/{mode_mmap_name}",
            "--tokenizer-model", tokenizer_model,
            "--tokenizer-type", "MMSentencePieceTokenizer",
            "--codec-type", codec_type,
            "--workers", "8",
            "--partitions", "1",
            "--instruction", instruction,
            "--instruction-dropout-rate", str(dropout),
            "--order", order,
            "--append-eod",
            "--quantizer-begin", str(quantizer_begin_idx),
            "--n-quantizer", str(num_quantizers),
            "--cot",
            "--use-token-level-interleave",
            "--use-audio-icl",
            "--audio-prompt-mode", mode,
            "--audio-prompt-len", str(prompt_len),
            "--keep-sequential-samples"
        ]
        
        # Запускаем команду
        run_command(cmd)
        
        # Удаляем временные файлы для этого режима
        remove_files(f"{data_root}/jsonl/{name_prefix}_*.jsonl")
        remove_files(f"{data_root}/{mode_mmap_name}_*_text_document.bin")
        remove_files(f"{data_root}/{mode_mmap_name}_*_text_document.idx")


def main():
    """
    Основная функция для запуска предобработки данных
    """
    parser = argparse.ArgumentParser(description='Preprocess data for YuE model training')
    parser.add_argument('data_setting', help='Data setting (e.g., dummy)')
    parser.add_argument('mode_type', help='Mode type (cot or icl_cot)')
    parser.add_argument('tokenizer_model', help='Path to the tokenizer model')
    parser.add_argument('--audio-prompt-modes', nargs='+', default=['dual', 'inst', 'vocal', 'mixture'],
                       help='Audio prompt modes for icl_cot (default: dual inst vocal mixture)')
    
    args = parser.parse_args()
    
    # Проверка обязательных параметров
    if not args.data_setting or not args.mode_type:
        print("Usage: python preprocess.py <setting> <mode_type> <tokenizer_model>")
        print("  <setting>: e.g., dummy")
        print("  <mode_type>: cot or icl_cot")
        sys.exit(1)
    
    # Общие настройки на основе DATA_SETTING
    if args.data_setting == "dummy":
        data_root = "example"
        name_prefix = "dummy.msa.xcodec_16k"
        codec_type = "xcodec"
        instruction = "Generate music from the given lyrics segment by segment."
        order = "textfirst"
        dropout = 0.0
        keep_sequential_samples = True
        quantizer_begin_idx = 0
        num_quantizers = 1
    else:
        print(f"Invalid setting: {args.data_setting}")
        sys.exit(1)
    
    # Выполнение в зависимости от режима
    if args.mode_type == "cot":
        preprocess_cot_mode(
            data_root, name_prefix, args.tokenizer_model, codec_type,
            instruction, order, dropout, quantizer_begin_idx,
            num_quantizers, keep_sequential_samples
        )
    elif args.mode_type == "icl_cot":
        preprocess_icl_cot_mode(
            data_root, name_prefix, args.tokenizer_model, codec_type,
            instruction, order, dropout, quantizer_begin_idx,
            num_quantizers, keep_sequential_samples, args.audio_prompt_modes
        )
    else:
        print(f"Invalid mode_type: {args.mode_type}. Use 'cot' or 'icl_cot'.")
        sys.exit(1)
    
    print(f"Preprocessing finished for setting '{args.data_setting}' and mode_type '{args.mode_type}'.")


if __name__ == "__main__":
    main() 