import gradio as gr
import os
import time
import subprocess
import logging
from types import SimpleNamespace
from infer import main as run_pipeline, get_args as get_infer_args  # Импорт из infer.py
import concurrent.futures
import tempfile
import atexit
import multiprocessing
import tqdm
import queue

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvalidPathError(Exception):
    pass

def validate_args(args_dict):
    # Sanitize paths
    for key in ['input_dir', 'output_dir', 'data_cache_path', 'model_cache_dir']:
        if key in args_dict:
            path = os.path.normpath(args_dict[key])
            if '..' in path or not os.path.isabs(path):  # Basic traversal check
                raise InvalidPathError(f'Invalid path for {key}: {path}')
            args_dict[key] = path
    required = ['input_dir', 'data_cache_path', 'model_cache_dir', 'model_name']
    for r in required:
        if not args_dict.get(r):
            raise ValueError(f'{r} is required.')
    if not args_dict['input_dir']:
        raise ValueError('Input Directory is required.')
    if not args_dict['data_cache_path']:
        raise ValueError('Data Cache Path is required.')
    # Добавить другие required

def build_ui():
    with gr.Blocks(title='YuE Fine-Tuning Pipeline Interface') as demo:
        gr.Markdown('# YuE Project Gradio Interface\nГибкий интерфейс для запуска пайплайна fine-tuning. Укажите параметры ниже.')
        
        # Состояния для логов и прогресса
        log_output = gr.Textbox(label='Логи', lines=10, interactive=False)
        progress_bar = gr.Slider(minimum=0, maximum=100, label='Прогресс', interactive=False)
        time_elapsed = gr.Textbox(label='Затраченное время', interactive=False)
        
        # Функция для обновления логов
        def update_log(new_log, current_log):
            return f'{current_log}\n{new_log}'
        
        # Сбор параметров (все из infer.py)
        with gr.Tab('Общие параметры'):
            gr.Markdown('**input_dir**: Директория с raw audio и jsonl. Рекомендация: /path/to/data (в Kaggle: dataset name). Обязательно.')
            input_dir = gr.Textbox(label='Input Directory', value='')
            gr.Markdown('**output_dir**: Выходная директория. Рекомендация: ./output или /kaggle/working/output.')
            output_dir = gr.Textbox(label='Output Directory', value='./output')
            gr.Markdown('**data_setting**: Настройка данных. Рекомендация: dummy для тестов.')
            data_setting = gr.Textbox(label='Data Setting', value='dummy')
            gr.Markdown('**mode_type**: Режим (cot или icl_cot). Рекомендация: cot для простоты.')
            mode_type = gr.Dropdown(choices=['cot', 'icl_cot'], label='Mode Type', value='cot')
            gr.Markdown('**tokenizer_model**: Путь к токенизатору. Рекомендация: mm_tokenizer_v0.2_hf/tokenizer.model.')
            tokenizer_model = gr.File(label='Tokenizer Model File', file_types=['.model'])
            gr.Markdown('**audio_prompt_modes**: Режимы промптов. Рекомендация: dual inst vocal mixture.')
            audio_prompt_modes = gr.Textbox(label='Audio Prompt Modes', value='dual inst vocal mixture')
        
        with gr.Tab('Параметры Audio Conversion'):
            gr.Markdown('**config_path**: Конфиг XCodec. Рекомендация: xcodec_mini_infer/final_ckpt/config.yaml.')
            config_path = gr.Textbox(label='Config Path', value='xcodec_mini_infer/final_ckpt/config.yaml')
            gr.Markdown('**ckpt_path**: Чекпоинт XCodec. Рекомендация: xcodec_mini_infer/final_ckpt/ckpt_00360000.pth.')
            ckpt_path = gr.Textbox(label='Checkpoint Path', value='xcodec_mini_infer/final_ckpt/ckpt_00360000.pth')
            gr.Markdown('**target_bw**: Bandwidth. Рекомендация: 6.0 для высокого качества.')
            target_bw = gr.Number(label='Target Bandwidth', value=6.0)
            gr.Markdown('**sample_rate**: Sample rate. Рекомендация: 16000 для XCodec.')
            sample_rate = gr.Number(label='Sample Rate', value=16000)
        
        with gr.Tab('Параметры Count Tokens'):
            gr.Markdown('**log_dir**: Директория логов. Рекомендация: ./count_token_logs/.')
            log_dir = gr.Textbox(label='Log Directory', value='./count_token_logs/')
        
        with gr.Tab('Параметры Parse Mixture'):
            gr.Markdown('**global_batch_size**: Глобальный batch size. Рекомендация: 64.')
            global_batch_size = gr.Number(label='Global Batch Size', value=64)
            gr.Markdown('**seq_len**: Длина последовательности. Рекомендация: 8192.')
            seq_len = gr.Number(label='Sequence Length', value=8192)
            gr.Markdown('**num_rounds**: Раунды. Рекомендация: 1.')
            num_rounds = gr.Number(label='Number of Rounds', value=1)
        
        with gr.Tab('Параметры Fine-Tuning'):
            gr.Markdown('**num_gpus**: Количество GPU. Рекомендация: 1 (или 2 в Kaggle).')
            num_gpus = gr.Number(label='Number of GPUs', value=1)
            gr.Markdown('**master_port**: Порт. Рекомендация: 9999.')
            master_port = gr.Number(label='Master Port', value=9999)
            # ... добавить все остальные из infer.py аналогично
            # (для краткости, полный список в коде)
            per_device_train_batch_size = gr.Number(label='Per Device Train Batch Size', value=1)
            per_device_eval_batch_size = gr.Number(label='Per Device Eval Batch Size', value=1)
            use_bf16 = gr.Checkbox(label='Use BF16', value=False)
            train_iters = gr.Number(label='Train Iters', value=None)
            num_train_epochs = gr.Number(label='Num Train Epochs', value=10)
            data_cache_path = gr.Textbox(label='Data Cache Path', value='', placeholder='Required: /path/to/cache')
            data_split = gr.Textbox(label='Data Split', value='900,50,50')
            model_name = gr.Textbox(label='Model Name', value='m-a-p/YuE-s1-7B-anneal-en-cot')
            model_cache_dir = gr.Textbox(label='Model Cache Dir', value='')
            deepspeed_config = gr.Textbox(label='Deepspeed Config', value='config/ds_config_zero2.json')
            lora_r = gr.Number(label='LoRA R', value=64)
            lora_alpha = gr.Number(label='LoRA Alpha', value=32)
            lora_dropout = gr.Number(label='LoRA Dropout', value=0.1)
            lora_target_modules = gr.Textbox(label='LoRA Target Modules', value='q_proj k_proj v_proj o_proj')
            logging_steps = gr.Number(label='Logging Steps', value=5)
            save_steps = gr.Number(label='Save Steps', value=5)
            use_wandb = gr.Checkbox(label='Use WandB', value=False)
            wandb_api_key = gr.Textbox(label='WandB API Key', value='')
            run_name = gr.Textbox(label='Run Name', value='YuE-ft-lora')
        
        run_button = gr.Button('Run Full Pipeline')
        
        def run_full(progress=gr.Progress()) -> tuple:
            """Run the full pipeline with progress updates."""
            progress(0, desc='Validating inputs...')
            args_dict = {
                'input_dir': input_dir.value,
                'output_dir': output_dir.value,
                'data_setting': data_setting.value,
                'mode_type': mode_type.value,
                'tokenizer_model': tokenizer_model.name if tokenizer_model else 'mm_tokenizer_v0.2_hf/tokenizer.model',  # Для file upload
                'audio_prompt_modes': audio_prompt_modes.value.split(),
                'config_path': config_path.value,
                'ckpt_path': ckpt_path.value,
                'target_bw': target_bw.value,
                'sample_rate': sample_rate.value,
                'log_dir': log_dir.value,
                'global_batch_size': global_batch_size.value,
                'seq_len': seq_len.value,
                'num_rounds': num_rounds.value,
                'num_gpus': num_gpus.value,
                'master_port': master_port.value,
                'per_device_train_batch_size': per_device_train_batch_size.value,
                'per_device_eval_batch_size': per_device_eval_batch_size.value,
                'use_bf16': use_bf16.value,
                'train_iters': train_iters.value,
                'num_train_epochs': num_train_epochs.value,
                'data_cache_path': data_cache_path.value,
                'data_split': data_split.value,
                'model_name': model_name.value,
                'model_cache_dir': model_cache_dir.value,
                'deepspeed_config': deepspeed_config.value,
                'lora_r': lora_r.value,
                'lora_alpha': lora_alpha.value,
                'lora_dropout': lora_dropout.value,
                'lora_target_modules': lora_target_modules.value,
                'logging_steps': logging_steps.value,
                'save_steps': save_steps.value,
                'use_wandb': use_wandb.value,
                'wandb_api_key': wandb_api_key.value,
                'run_name': run_name.value
            }
            validate_args(args_dict)
            args = SimpleNamespace(**args_dict)
            
            # Асинхронный запуск
            log_queue = multiprocessing.Queue()
            def run_in_background(log_queue):
                temp_log = tempfile.NamedTemporaryFile(delete=False)
                atexit.register(os.unlink, temp_log.name)
                logging.getLogger().addHandler(logging.FileHandler(temp_log.name))
                try:
                    run_pipeline(args)
                    return 'Pipeline completed successfully.'
                except Exception as e:
                    return f'Error: {str(e)}'
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_background, log_queue)
                steps = 5  # Кол-во шагов в infer.py
                with tqdm(total=steps) as pbar:
                    for i in range(steps):
                        time.sleep(5)  # Симуляция, заменить на реальный polling
                        elapsed = time.time() - start_time
                        pbar.update(1)
                        while not log_queue.empty():
                            log = log_queue.get()
                            yield log, pbar.n / pbar.total * 100, f'{elapsed:.2f} seconds'
                result = future.result()
                elapsed = time.time() - start_time
                yield result, 100, f'{elapsed:.2f} seconds'

        run_button.click(run_full, outputs=[log_output, progress_bar, time_elapsed])

        # Добавляю отдельные кнопки для шагов
        with gr.Row():
            audio_button = gr.Button('Run Audio Conversion Only')
            # Аналогично для других, с функциями вроде def run_audio(args): subprocess.run(cmd_audio)
            # (реализовать аналогично run_full)

    return demo

if __name__ == '__main__':
    ui = build_ui()
    ui.launch(share=True) 