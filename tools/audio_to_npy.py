import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from models.soundstream_hubert_new import SoundStream  # Из xcodec_mini_infer
import logging
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Convert raw audio to .npy codec files using XCodec for high quality.')
    group = parser.add_argument_group('Input/Output')
    group.add_argument('--input_dir', type=str, required=True, default='/kaggle/input/data', help='Directory with vocal.wav, instrumental.wav, mix.wav')
    group.add_argument('--output_dir', type=str, required=True, default='/kaggle/working/FINETUNE-YUE/npy', help='Output directory for .npy files')
    parser.add_argument('--config_path', type=str, default='xcodec_mini_infer/final_ckpt/config.yaml', help='Path to XCodec config')
    parser.add_argument('--ckpt_path', type=str, default='xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to XCodec checkpoint')
    parser.add_argument('--target_bw', type=float, default=6.0, help='Target bandwidth for encoding (higher for quality)')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Target sample rate')
    return parser.parse_args()

def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = SoundStream(**config.generator.config)  # Адаптируйте под вашу модель
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['codec_model'])
    model.eval()
    return model

def encode_audio(model, audio_path, target_bw, sample_rate):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Mono
    waveform = waveform.unsqueeze(0)  # Batch dim
    with torch.no_grad():
        codes = model.encode(waveform, target_bw=target_bw)
    return codes.cpu().numpy().astype(np.int16)  # Для совместимости с CodecManipulator

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.config_path, args.ckpt_path)

    audio_types = ['vocal', 'instrumental', 'mix']
    for audio_type in tqdm(audio_types, desc='Processing audio types'):
        input_path = os.path.join(args.input_dir, f'{audio_type}.wav')  # Предполагаем WAV, добавьте MP3 если нужно
        if not os.path.exists(input_path):
            logger.warning(f'Warning: {input_path} not found, skipping.')
            continue
        logger.info(f'Encoding {audio_type} from {input_path}...')
        codes = encode_audio(model, input_path, args.target_bw, args.sample_rate)
        output_path = os.path.join(args.output_dir, f'{audio_type}.npy')
        if os.path.exists(output_path):
            logger.info(f'Cached {audio_type} codes from {output_path}')
            continue
        np.save(output_path, codes)
        logger.info(f'Saved {audio_type} codes to {output_path}')

if __name__ == '__main__':
    main() 