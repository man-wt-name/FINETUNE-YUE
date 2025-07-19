import os
import argparse
import subprocess
import logging
import yaml
import torch
from pathlib import Path
from typing import Any, Dict
from core.exceptions import InvalidInputError

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    pass

def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Главный скрипт для полного пайплайна fine-tuning YuE модели в Kaggle или локально.')
    
    # Общие параметры
    parser.add_argument('--input_dir', type=str, required=True, help='Директория с входными данными (raw audio и jsonl для Kaggle: /kaggle/input).')
    parser.add_argument('--output_dir', type=str, default='./output', help='Выходная директория (для Kaggle: /kaggle/working).')
    parser.add_argument('--data_setting', type=str, default='dummy', help='Настройка данных (e.g., dummy).')
    parser.add_argument('--mode_type', type=str, choices=['cot', 'icl_cot'], default='cot', help='Тип режима (cot or icl_cot).')
    parser.add_argument('--tokenizer_model', type=str, default='mm_tokenizer_v0.2_hf/tokenizer.model', help='Путь к модели токенизатора (по умолчанию: mm_tokenizer_v0.2_hf/tokenizer.model).')
    parser.add_argument('--audio_prompt_modes', nargs='+', default=['dual', 'inst', 'vocal', 'mixture'], help='Режимы аудио-промптов.')
    
    # Параметры для audio_to_npy
    parser.add_argument('--config_path', type=str, default='xcodec_mini_infer/final_ckpt/config.yaml', help='Путь к конфигу XCodec.')
    parser.add_argument('--ckpt_path', type=str, default='xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Путь к чекпоинту XCodec.')
    parser.add_argument('--target_bw', type=float, default=6.0, help='Целевая полоса пропускания для кодирования.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Частота дискретизации (адаптировано для XCodec 16kHz).')
    
    # Параметры для count_tokens
    parser.add_argument('--log_dir', type=str, default='./count_token_logs/', help='Директория для логов подсчета токенов.')
    
    # Параметры для parse_mixture (генерируем YAML)
    parser.add_argument('--global_batch_size', type=int, default=64, help='Глобальный размер батча.')
    parser.add_argument('--seq_len', type=int, default=8192, help='Длина последовательности.')
    parser.add_argument('--num_rounds', type=int, default=1, help='Количество раундов повторения датасета.')
    
    # Параметры для run_finetune
    parser.add_argument('--num_gpus', type=int, default=1, help='Количество GPU (в Kaggle обычно 2).')
    parser.add_argument('--master_port', type=int, default=9999, help='Мастер-порт для distributed.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='Размер батча на устройство для тренировки.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Размер батча на устройство для оценки.')
    parser.add_argument('--use_bf16', action='store_true', help='Использовать bf16.')
    parser.add_argument('--train_iters', type=int, default=None, help='Количество итераций тренировки (будет перезаписано parse_mixture).')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Количество эпох тренировки.')
    parser.add_argument('--data_cache_path', type=str, required=True, help='Путь к кэшу данных.')
    parser.add_argument('--data_split', type=str, default='900,50,50', help='Разделение данных.')
    parser.add_argument('--model_name', type=str, default='m-a-p/YuE-s1-7B-anneal-en-cot', help='Имя или путь к базовой модели.')
    parser.add_argument('--model_cache_dir', type=str, required=True, help='Кэш-директория модели.')
    parser.add_argument('--deepspeed_config', type=str, default='config/ds_config_zero2.json', help='Путь к DeepSpeed конфигу.')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout.')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj k_proj v_proj o_proj', help='LoRA target modules.')
    parser.add_argument('--logging_steps', type=int, default=5, help='Шаги логирования.')
    parser.add_argument('--save_steps', type=int, default=5, help='Шаги сохранения.')
    parser.add_argument('--use_wandb', action='store_true', help='Использовать WandB.')
    parser.add_argument('--wandb_api_key', type=str, default='', help='WandB API key.')
    parser.add_argument('--run_name', type=str, default='YuE-ft-lora', help='Имя запуска для WandB.')
    
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        raise InvalidInputError('Invalid input_dir')
    return args

def is_kaggle() -> bool:
    """Check if running in Kaggle environment."""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def adjust_paths_for_kaggle(args):
    if is_kaggle():
        logger.info('Обнаружена среда Kaggle. Адаптирую пути.')
        args.input_dir = os.path.normpath('/kaggle/input/' + os.path.basename(args.input_dir))
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f'Kaggle input dir not found: {args.input_dir}')
        args.output_dir = '/kaggle/working/FINETUNE-YUE/output'
        args.data_cache_path = '/kaggle/working/FINETUNE-YUE/cache'
        args.model_cache_dir = '/kaggle/working/FINETUNE-YUE/model_cache'
        args.log_dir = '/kaggle/working/FINETUNE-YUE/logs'
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.data_cache_path, exist_ok=True)
        os.makedirs(args.model_cache_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    return args

def run_step(cmd, step_name):
    logger.info(f'Запуск шага: {step_name}')
    logger.info(f'Команда: {" ".join(cmd)}')
    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f'Ошибка в шаге {step_name}: {process.stderr}')
            raise PipelineError(f'Шаг {step_name} провалился.')
        logger.info(f'Шаг {step_name} завершен успешно. Вывод: {process.stdout}')
        return process.stdout
    except Exception as e:
        raise PipelineError(f'Error in {step_name}: {e}')

def main():
    """Main function for pipeline."""
    args = get_args()
    args = adjust_paths_for_kaggle(args)
    
    # Шаг 1: Конвертация аудио в .npy
    npy_output = os.path.join(args.output_dir, 'npy')
    cmd_audio = [
        'python', 'tools/audio_to_npy.py',
        '--input_dir', args.input_dir,
        '--output_dir', npy_output,
        '--config_path', args.config_path,
        '--ckpt_path', args.ckpt_path,
        '--target_bw', str(args.target_bw),
        '--sample_rate', str(args.sample_rate)
    ]
    run_step(cmd_audio, 'Конвертация аудио в .npy')
    
    # Шаг 2: Препроцессинг данных (предполагаем jsonl в input_dir/jsonl, npy в npy_output)
    jsonl_path = os.path.join(args.input_dir, f'jsonl/dummy.msa.xcodec_16k.jsonl')  # Адаптировать на основе data_setting
    mmap_output = os.path.join(args.output_dir, 'mmap')
    cmd_preprocess = [
        'python', 'scripts/preprocess_data.py',
        args.data_setting,
        args.mode_type,
        args.tokenizer_model,
        '--audio_prompt_modes'
    ] + args.audio_prompt_modes
    # Модифицировать для использования npy_output как data_root, но скрипт использует fixed paths; предположим патч в subprocess
    run_step(cmd_preprocess, 'Препроцессинг данных')
    
    # Шаг 3: Подсчет токенов (input: mmap_output)
    cmd_count = [
        'python', 'scripts/count_tokens.py',
        '--parent_dir', mmap_output,
        '--log_dir', args.log_dir
    ]
    run_step(cmd_count, 'Подсчет токенов')
    
    # Шаг 4: Парсинг смеси (создаем временный YAML)
    mixture_cfg = {
        'TOKEN_COUNT_LOG_DIR': args.log_dir,
        'GLOBAL_BATCH_SIZE': args.global_batch_size,
        'SEQ_LEN': args.seq_len,
        '1_ROUND': args.num_rounds  # Пример, адаптировать по cfg
    }
    temp_yaml = os.path.join(args.output_dir, 'temp_mixture_cfg.yml')
    with open(temp_yaml, 'w') as f:
        yaml.dump(mixture_cfg, f)
    cmd_parse = ['python', 'core/parse_mixture.py', '-c', temp_yaml]
    parse_output = run_step(cmd_parse, 'Парсинг смеси данных')
    # Извлечь DATA_PATH и TRAIN_ITERS из вывода
    data_path = [line.split(': ')[1] for line in parse_output.split('\n') if line.startswith('DATA_PATH')][0]
    train_iters = int([line.split(': ')[1] for line in parse_output.split('\n') if line.startswith('TRAIN_ITERS')][0])
    args.train_iters = train_iters
    
    # Шаг 5: Запуск fine-tuning
    cmd_finetune = [
        'python', 'scripts/run_finetune.py',
        # Передача всех параметров через аргументы, но скрипт не имеет argparse; предположим использование в subprocess с модификацией env или патчем
    ]
    # Поскольку run_finetune.py имеет hardcoded, для простоты предполагаем, что он адаптирован или используем os.environ для overrides
    os.environ['DATA_PATH'] = data_path
    os.environ['TRAIN_ITERS'] = str(train_iters)
    # ... другие env vars
    run_step(cmd_finetune, 'Запуск fine-tuning')
    
    logger.info('Полный пайплайн завершен.')

if __name__ == '__main__':
    main() 