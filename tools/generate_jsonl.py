import os
import json
import argparse
import logging
from typing import List, Dict, Optional
import csv
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_jsonl(npy_dir: str, output_jsonl: str, prompt_template: str = 'Generate {type} audio', types: List[str] = ['mixture', 'vocal', 'instrumental'], csv_file: Optional[str] = None, add_duration: bool = False, sample_rate: int = 16000, type_mapping_str: str = '') -> None:
    """Generate JSONL file from .npy files in directory.

    Args:
        npy_dir: Directory with .npy files.
        output_jsonl: Output JSONL path.
        prompt_template: Template for prompts.
        types: Possible audio types.
    """
    if not os.path.isdir(npy_dir):
        raise FileNotFoundError(f'Directory not found: {npy_dir}')

    type_map: Dict[str, str] = {}
    if type_mapping_str:
        for pair in type_mapping_str.split(','):
            if '=' in pair:
                k, v = pair.split('=')
                type_map[k.strip()] = v.strip()
            else:
                logger.warning(f'Invalid mapping: {pair}')
    prompt_map: Dict[str, str] = {}
    if csv_file:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f'CSV not found: {csv_file}')
        with open(csv_file, 'r') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                if 'filename' in row and 'prompt' in row:
                    prompt_map[row['filename']] = row['prompt']

    with open(output_jsonl, 'w') as f:
        for file in os.listdir(npy_dir):
            if file.endswith('.npy'):
                # Type determination
                base = file.lower().replace('.npy', '')
                audio_type = type_map.get(base, 'mixture' if 'mix' in base else 'vocal' if 'vocal' in base else 'instrumental')
                # Prompt
                prompt = prompt_map.get(file, prompt_template.format(type=audio_type))
                entry = {
                    'audio_path': os.path.join('npy', file),
                    'prompt': prompt,
                    'type': audio_type
                }
                # Duration
                if add_duration:
                    npy_path = os.path.join(npy_dir, file)
                    codes = np.load(npy_path)
                    entry['duration'] = len(codes) / sample_rate
                f.write(json.dumps(entry) + '\n')
                logger.info(f'Added entry for {file}')
    logger.info(f'Generated {output_jsonl}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate JSONL from .npy files.')
    parser.add_argument('--npy_dir', required=True, help='Directory with .npy files.')
    parser.add_argument('--output_jsonl', default='data.jsonl', help='Output JSONL file.')
    parser.add_argument('--prompt_template', default='Generate {type} audio', help='Prompt template.')
    parser.add_argument('--csv_file', default=None, help='CSV file with filename,prompt mapping.')
    parser.add_argument('--add_duration', action='store_true', help='Add duration field from .npy.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for duration calc.')
    parser.add_argument('--type_mapping', default='', help='Type mapping like dummy=mix,vocals=vocal.')
    args = parser.parse_args()
    generate_jsonl(args.npy_dir, args.output_jsonl, args.prompt_template, args.csv_file, args.add_duration, args.sample_rate, args.type_mapping) 