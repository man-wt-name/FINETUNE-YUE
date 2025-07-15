import os
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir', type=str, default='/workspace/dataset/music', help='Parent directory to search for .bin files')
    parser.add_argument('--log_dir', type=str, default='./count_token_logs/', help='Directory for log files')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # Find all .bin files
    bins = [os.path.join(root, file) for root, _, files in os.walk(args.parent_dir) for file in files if file.endswith('.bin')]

    def process_bin(mmap_path):
        mmap_size = os.path.getsize(mmap_path) / (1024 ** 3)  # Size in GB
        print(f'Counting mmap file: {mmap_path}, size: {mmap_size:.2f} GB')

        subdir = mmap_path.replace(args.parent_dir + '/', '').replace('/', '_')
        log_path = os.path.join(args.log_dir, f'count.{subdir}.log')

        with open(log_path, 'w') as log_file:
            subprocess.run(['python', 'tools/count_mmap_token.py', '--mmap_path', mmap_path], stdout=log_file, stderr=log_file)
        print(f'Finished processing: {mmap_path}')

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap(process_bin, bins), total=len(bins)))

if __name__ == '__main__':
    main() 