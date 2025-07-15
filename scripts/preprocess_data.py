import os
import argparse
import subprocess
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_setting', type=str, help='e.g., dummy')
    parser.add_argument('mode_type', type=str, choices=['cot', 'icl_cot'], help='cot or icl_cot')
    parser.add_argument('tokenizer_model', type=str, default='mm_tokenizer_v0.2_hf/tokenizer.model', nargs='?', help='Путь к модели токенизатора (по умолчанию: mm_tokenizer_v0.2_hf/tokenizer.model).')
    parser.add_argument('--audio_prompt_modes', nargs='+', default=['dual', 'inst', 'vocal', 'mixture'])
    return parser.parse_args()

def main():
    args = get_args()
    data_setting = args.data_setting
    mode_type = args.mode_type
    tokenizer_model = args.tokenizer_model if args.tokenizer_model else 'mm_tokenizer_v0.2_hf/tokenizer.model'
    audio_prompt_modes = args.audio_prompt_modes

    if data_setting == 'dummy':
        data_root = 'example'
        name_prefix = 'dummy.msa.xcodec_16k'
        codec_type = 'xcodec'
        instruction = 'Generate music from the given lyrics segment by segment.'
        order = 'textfirst'
        dropout = 0.0
        keep_sequential_samples = True
        quantizer_begin_idx = 0
        num_quantizers = 1
    else:
        raise ValueError(f'Invalid setting: {data_setting}')

    jsonl_name = f'jsonl/{name_prefix}.jsonl'

    if mode_type == 'cot':
        print("Running in 'cot' mode...")
        name_suffix = 'stage_1_token_level_interleave_cot_xcodec'
        mmap_name = f'mmap/{name_prefix}_{name_suffix}_{order}'

        os.makedirs(os.path.join(data_root, mmap_name), exist_ok=True)

        cmd = [
            'python', 'core/preprocess_data_conditional_xcodec_segment.py',
            '--input', os.path.join(data_root, jsonl_name),
            '--output-prefix', os.path.join(data_root, mmap_name),
            '--tokenizer-model', tokenizer_model,
            '--tokenizer-type', 'MMSentencePieceTokenizer',
            '--codec-type', codec_type,
            '--workers', '8',
            '--partitions', '1',
            '--instruction', instruction,
            '--instruction-dropout-rate', str(dropout),
            '--order', order,
            '--append-eod',
            '--quantizer-begin', str(quantizer_begin_idx),
            '--n-quantizer', str(num_quantizers),
            '--use-token-level-interleave',
            '--keep-sequential-samples',
            '--cot'
        ]
        print(' '.join(cmd))
        time.sleep(5)
        subprocess.run(cmd)

        # Cleanup
        for f in os.listdir(os.path.join(data_root, 'jsonl')):
            if f.startswith(name_prefix) and f.endswith('.jsonl'):
                os.remove(os.path.join(data_root, 'jsonl', f))
        for f in os.listdir(os.path.join(data_root, mmap_name)):
            if f.endswith('_text_document.bin') or f.endswith('_text_document.idx'):
                os.remove(os.path.join(data_root, mmap_name, f))

    elif mode_type == 'icl_cot':
        print("Running in 'icl_cot' mode...")
        name_suffix = 'stage_1_token_level_interleave_long_prompt_msa'
        mmap_name = f'mmap/{name_prefix}_{name_suffix}_{order}'
        prompt_len = 30

        os.makedirs(os.path.join(data_root, mmap_name), exist_ok=True)

        for mode in audio_prompt_modes:
            print(f'Processing mode: {mode}')
            mode_mmap_name = f'{mmap_name}_{mode}'
            os.makedirs(os.path.join(data_root, mode_mmap_name), exist_ok=True)

            cmd = [
                'python', 'core/preprocess_data_conditional_xcodec_segment.py',
                '--input', os.path.join(data_root, jsonl_name),
                '--output-prefix', os.path.join(data_root, mode_mmap_name),
                '--tokenizer-model', tokenizer_model,
                '--tokenizer-type', 'MMSentencePieceTokenizer',
                '--codec-type', codec_type,
                '--workers', '8',
                '--partitions', '1',
                '--instruction', instruction,
                '--instruction-dropout-rate', str(dropout),
                '--order', order,
                '--append-eod',
                '--quantizer-begin', str(quantizer_begin_idx),
                '--n-quantizer', str(num_quantizers),
                '--cot',
                '--use-token-level-interleave',
                '--use-audio-icl',
                '--audio-prompt-mode', mode,
                '--audio-prompt-len', str(prompt_len),
                '--keep-sequential-samples'
            ]
            print(' '.join(cmd))
            time.sleep(5)
            subprocess.run(cmd)

            # Cleanup
            for f in os.listdir(os.path.join(data_root, 'jsonl')):
                if f.startswith(name_prefix) and f.endswith('.jsonl'):
                    os.remove(os.path.join(data_root, 'jsonl', f))
            for f in os.listdir(os.path.join(data_root, mode_mmap_name)):
                if f.endswith('_text_document.bin') or f.endswith('_text_document.idx'):
                    os.remove(os.path.join(data_root, mode_mmap_name, f))

    print(f'Preprocessing finished for setting "{data_setting}" and mode_type "{mode_type}".')

if __name__ == '__main__':
    main() 