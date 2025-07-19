import os
import argparse
import subprocess
import sys
import logging
from typing import List, Optional
from core.exceptions import InvalidInputError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def print_help() -> None:
    """Print help message for the script."""
    logger.info('======================================================')
    logger.info('YuE Fine-tuning Script Help')
    logger.info('======================================================')
    logger.info('Before running this script, please update the following variables:')
    logger.info('1. Data paths:')
    logger.info('   DATA_PATH - Replace <weight_and_path_to_data_X> with actual weights and data paths')
    logger.info('   DATA_CACHE_PATH - Replace <path_to_data_cache> with actual cache directory')
    logger.info('2. Model configuration:')
    logger.info('   TOKENIZER_MODEL_PATH - Replace <path_to_tokenizer_model> with actual tokenizer path')
    logger.info('   MODEL_CACHE_DIR - Replace <path_to_model_cache> with actual cache directory')
    logger.info('   OUTPUT_DIR - Replace <path_to_output_dir> with actual output directory')
    logger.info('3. If using WandB:')
    logger.info('   WANDB_API_KEY - Replace <your_wandb_api_key> with your actual API key')
    logger.info('Example usage: python scripts/run_finetune.py with config updates in code')
    logger.info('======================================================')
    sys.exit(1)

def check_placeholders(data_path: str, data_cache_path: str, tokenizer_model_path: str, model_cache_dir: str, output_dir: str, use_wandb: bool, wandb_api_key: str) -> None:
    """Check for placeholders in configuration."""
    has_placeholders = False
    data_path = os.path.normpath(data_path)
    if '<' in data_path:
        raise InvalidInputError('Placeholder in path')
    if '<weight_and_path_to_data' in data_path:
        logger.error('Error: Please set actual weight and data paths in DATA_PATH variable.')
        has_placeholders = True
    if '<path_to_data_cache>' in data_cache_path:
        logger.error('Error: Please set actual data cache path in DATA_CACHE_PATH variable.')
        has_placeholders = True
    if '<path_to_tokenizer_model>' in tokenizer_model_path:
        logger.error('Error: Please set actual tokenizer model path in TOKENIZER_MODEL_PATH variable.')
        has_placeholders = True
    if '<path_to_model_cache>' in model_cache_dir:
        logger.error('Error: Please set actual model cache directory in MODEL_CACHE_DIR variable.')
        has_placeholders = True
    if '<path_to_output_dir>' in output_dir:
        logger.error('Error: Please set actual output directory in OUTPUT_DIR variable.')
        has_placeholders = True
    if use_wandb and '<your_wandb_api_key>' in wandb_api_key:
        logger.error('Error: Please set actual WandB API key in WANDB_API_KEY variable or disable WandB.')
        has_placeholders = True
    if has_placeholders:
        logger.warning('\nPlease update the script with your actual paths and values.')
        logger.warning("Run 'python scripts/run_finetune.py --help' for more information.")
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--help', '-h', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    if args.help:
        print_help()

    # Чтение всех ключевых параметров из переменных окружения (если не задано — используется дефолт)
    try:
        num_gpus = int(os.environ.get('NUM_GPUS', 8))
    except ValueError:
        raise InvalidInputError('Invalid NUM_GPUS')
    master_port = int(os.environ.get('MASTER_PORT', 9999))
    per_device_train_batch_size = int(os.environ.get('PER_DEVICE_TRAIN_BATCH_SIZE', 1))
    per_device_eval_batch_size = int(os.environ.get('PER_DEVICE_EVAL_BATCH_SIZE', 1))
    global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', num_gpus * per_device_train_batch_size))
    use_bf16 = os.environ.get('USE_BF16', 'True').lower() in ['1', 'true', 'yes']
    seq_length = int(os.environ.get('SEQ_LENGTH', 8192))
    train_iters = int(os.environ.get('TRAIN_ITERS', 150))
    num_train_epochs = int(os.environ.get('NUM_TRAIN_EPOCHS', 10))

    data_path = os.environ.get('DATA_PATH', '<weight_and_path_to_data_X>')
    data_cache_path = os.environ.get('DATA_CACHE_PATH', '/kaggle/working/FINETUNE-YUE/cache')
    data_split = os.environ.get('DATA_SPLIT', '900,50,50')

    tokenizer_model_path = os.environ.get('TOKENIZER_MODEL_PATH', 'mm_tokenizer_v0.2_hf/tokenizer.model')
    model_name = os.environ.get('MODEL_NAME', 'm-a-p/YuE-s1-7B-anneal-en-cot')
    model_cache_dir = os.environ.get('MODEL_CACHE_DIR', '/kaggle/working/FINETUNE-YUE/model_cache')
    output_dir = os.environ.get('OUTPUT_DIR', '/kaggle/working/FINETUNE-YUE/output')
    deepspeed_config = os.environ.get('DEEPSPEED_CONFIG', 'config/ds_config_zero2.json')

    lora_r = int(os.environ.get('LORA_R', 64))
    lora_alpha = int(os.environ.get('LORA_ALPHA', 32))
    lora_dropout = float(os.environ.get('LORA_DROPOUT', 0.1))
    lora_target_modules = os.environ.get('LORA_TARGET_MODULES', 'q_proj k_proj v_proj o_proj')

    logging_steps = int(os.environ.get('LOGGING_STEPS', 5))
    save_steps = int(os.environ.get('SAVE_STEPS', 5))
    use_wandb = os.environ.get('USE_WANDB', 'True').lower() in ['1', 'true', 'yes']
    wandb_api_key = os.environ.get('WANDB_API_KEY', '<your_wandb_api_key>')
    run_name = os.environ.get('RUN_NAME', 'YuE-ft-lora')

    # Comment out or modify check_placeholders to warn
    # check_placeholders(data_path, data_cache_path, tokenizer_model_path, model_cache_dir, output_dir, use_wandb, wandb_api_key)
    if '<' in data_path or '<' in data_cache_path:  # Simple check
        logger.warning('Warning: Some paths may need setting.')

    os.environ['WANDB_API_KEY'] = wandb_api_key
    os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), os.environ.get('PYTHONPATH', ''))

    logger.info('===============================================')
    logger.info('YuE Fine-tuning Configuration:')
    logger.info('===============================================')
    logger.info(f'Number of GPUs: {num_gpus}')
    logger.info(f'Global batch size: {global_batch_size}')
    logger.info(f'Model: {model_name}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Training epochs: {num_train_epochs}')
    logger.info('===============================================')

    cmd = [
        'torchrun', '--nproc_per_node', str(num_gpus), '--master_port', str(master_port),
        'scripts/train_lora.py',
        '--seq-length', str(seq_length),
        '--data-path', data_path,
        '--data-cache-path', data_cache_path,
        '--split', data_split,
        '--tokenizer-model', tokenizer_model_path,
        '--global-batch-size', str(global_batch_size),
        '--per-device-train-batch-size', str(per_device_train_batch_size),
        '--per-device-eval-batch-size', str(per_device_eval_batch_size),
        '--train-iters', str(train_iters),
        '--num-train-epochs', str(num_train_epochs),
        '--logging-steps', str(logging_steps),
        '--save-steps', str(save_steps),
        '--deepspeed', deepspeed_config,
        '--model-name-or-path', model_name,
        '--cache-dir', model_cache_dir,
        '--output-dir', output_dir,
        '--lora-r', str(lora_r),
        '--lora-alpha', str(lora_alpha),
        '--lora-dropout', str(lora_dropout),
        '--lora-target-modules', *lora_target_modules.split()
    ]

    if use_wandb:
        cmd.extend(['--report-to', 'wandb', '--run-name', run_name])
    else:
        cmd.extend(['--report-to', 'none'])

    if use_bf16:
        cmd.append('--bf16')

    logger.info('Running command: ' + ' '.join(cmd))
    logger.info('===============================================')

    process = subprocess.run(cmd)

    if process.returncode == 0:
        logger.info('===============================================')
        logger.info('Fine-tuning completed successfully!')
        logger.info(f'Output saved to: {output_dir}')
        logger.info('===============================================')
    else:
        logger.error('===============================================')
        logger.error(f'Error: Fine-tuning failed with exit code {process.returncode}')
        logger.error('===============================================')
        sys.exit(1)

if __name__ == '__main__':
    main() 