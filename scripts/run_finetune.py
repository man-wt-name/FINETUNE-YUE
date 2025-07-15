import os
import argparse
import subprocess
import sys

def print_help():
    print('======================================================')
    print('YuE Fine-tuning Script Help')
    print('======================================================')
    print('Before running this script, please update the following variables:')
    print('1. Data paths:')
    print('   DATA_PATH - Replace <weight_and_path_to_data_X> with actual weights and data paths')
    print('   DATA_CACHE_PATH - Replace <path_to_data_cache> with actual cache directory')
    print('2. Model configuration:')
    print('   TOKENIZER_MODEL_PATH - Replace <path_to_tokenizer_model> with actual tokenizer path')
    print('   MODEL_CACHE_DIR - Replace <path_to_model_cache> with actual cache directory')
    print('   OUTPUT_DIR - Replace <path_to_output_dir> with actual output directory')
    print('3. If using WandB:')
    print('   WANDB_API_KEY - Replace <your_wandb_api_key> with your actual API key')
    print('Example usage: python scripts/run_finetune.py with config updates in code')
    print('======================================================')
    sys.exit(1)

def check_placeholders(data_path, data_cache_path, tokenizer_model_path, model_cache_dir, output_dir, use_wandb, wandb_api_key):
    has_placeholders = False
    if '<weight_and_path_to_data' in data_path:
        print('Error: Please set actual weight and data paths in DATA_PATH variable.')
        has_placeholders = True
    if '<path_to_data_cache>' in data_cache_path:
        print('Error: Please set actual data cache path in DATA_CACHE_PATH variable.')
        has_placeholders = True
    if '<path_to_tokenizer_model>' in tokenizer_model_path:
        print('Error: Please set actual tokenizer model path in TOKENIZER_MODEL_PATH variable.')
        has_placeholders = True
    if '<path_to_model_cache>' in model_cache_dir:
        print('Error: Please set actual model cache directory in MODEL_CACHE_DIR variable.')
        has_placeholders = True
    if '<path_to_output_dir>' in output_dir:
        print('Error: Please set actual output directory in OUTPUT_DIR variable.')
        has_placeholders = True
    if use_wandb and '<your_wandb_api_key>' in wandb_api_key:
        print('Error: Please set actual WandB API key in WANDB_API_KEY variable or disable WandB.')
        has_placeholders = True
    if has_placeholders:
        print('\nPlease update the script with your actual paths and values.')
        print("Run 'python scripts/run_finetune.py --help' for more information.")
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--help', '-h', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    if args.help:
        print_help()

    # Configuration Parameters
    num_gpus = 8
    master_port = 9999
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # Uncomment if needed

    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    global_batch_size = num_gpus * per_device_train_batch_size
    use_bf16 = True
    seq_length = 8192
    train_iters = 150
    num_train_epochs = 10

    data_path = '<weight_and_path_to_data_X>'
    data_cache_path = '<path_to_tokenizer_model>'
    data_split = '900,50,50'

    tokenizer_model_path = 'mm_tokenizer_v0.2_hf/tokenizer.model'
    model_name = 'm-a-p/YuE-s1-7B-anneal-en-cot'
    model_cache_dir = '<path_to_model_cache>'
    output_dir = '<path_to_output_dir>'
    deepspeed_config = 'config/ds_config_zero2.json'

    lora_r = 64
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = 'q_proj k_proj v_proj o_proj'

    logging_steps = 5
    save_steps = 5
    use_wandb = True
    wandb_api_key = '<your_wandb_api_key>'
    run_name = 'YuE-ft-lora'

    check_placeholders(data_path, data_cache_path, tokenizer_model_path, model_cache_dir, output_dir, use_wandb, wandb_api_key)

    os.environ['WANDB_API_KEY'] = wandb_api_key
    os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), os.environ.get('PYTHONPATH', ''))

    print('===============================================')
    print('YuE Fine-tuning Configuration:')
    print('===============================================')
    print(f'Number of GPUs: {num_gpus}')
    print(f'Global batch size: {global_batch_size}')
    print(f'Model: {model_name}')
    print(f'Output directory: {output_dir}')
    print(f'Training epochs: {num_train_epochs}')
    print('===============================================')

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

    print('Running command: ' + ' '.join(cmd))
    print('===============================================')

    process = subprocess.run(cmd)

    if process.returncode == 0:
        print('===============================================')
        print('Fine-tuning completed successfully!')
        print(f'Output saved to: {output_dir}')
        print('===============================================')
    else:
        print('===============================================')
        print(f'Error: Fine-tuning failed with exit code {process.returncode}')
        print('===============================================')
        sys.exit(1)

if __name__ == '__main__':
    main() 