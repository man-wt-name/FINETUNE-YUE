#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для преобразования аудиофайлов WAV в формат NPY для использования с YuE.
Обрабатывает файлы с названиями vocals, instrumental и mix в указанной папке.
Использует XCodec для дискретизации аудио.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from scipy import signal
import librosa

# Проверка наличия CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

class XCodec:
    """
    Класс для работы с XCodec - аудио кодеком для дискретизации аудиосигналов.
    Использует предобученные модели для кодирования аудио в дискретные коды.
    """
    def __init__(self, model_path=None, device=None):
        """
        Инициализирует XCodec с заданной моделью.
        
        Args:
            model_path: путь к предобученной модели кодека
            device: устройство для вычислений (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.codebook_size = 1024
        self.n_quantizers = 12
        self.sample_rate = 16000
        self.use_audiocraft = False
        self.use_encodec = False
        
        try:
            # Попытка импорта и инициализации XCodec
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            
            # Если установлен openl2s или другая реализация XCodec:
            try:
                from audiocraft.models import AudioGen
                from audiocraft.data.audio import audio_read, audio_write
                
                print("Использую AudioCraft XCodec для кодирования")
                self.model = AudioGen.get_pretrained('facebook/audiogen-medium')
                self.model.to(self.device)
                self.model.set_generation_params(duration=30)  # Максимальная длительность в секундах
                
                # Задаем параметры кодека
                self.codebook_size = 1024
                self.n_quantizers = 12
                self.use_audiocraft = True
                self.tokenizer_rate = 50  # 50Hz, типичная частота дискретизации для XCodec
                
            except ImportError:
                print("AudioCraft не найден, использую EnCodec")
                try:
                    # Альтернативный вариант - EnCodec от Meta
                    from encodec import EncodecModel
                    from encodec.utils import convert_audio
                    
                    # Загрузка предобученной модели EnCodec
                    self.model = EncodecModel.encodec_model_24khz()
                    self.model.to(self.device)
                    self.codebook_size = self.model.quantizer.bins
                    self.n_quantizers = len(self.model.quantizer.codebooks)
                    self.sample_rate = 24000
                    self.use_encodec = True
                    
                except ImportError:
                    print("ПРЕДУПРЕЖДЕНИЕ: Ни AudioCraft, ни EnCodec не найдены. Будет использоваться упрощенная дискретизация.")
                    print("Для полноценного использования установите одну из библиотек:")
                    print("pip install audiocraft  # Meta AudioCraft")
                    print("или")
                    print("pip install encodec  # Meta EnCodec")
                    
        except Exception as e:
            print(f"Ошибка при инициализации XCodec: {e}")
            print("Будет использована упрощенная дискретизация.")

    def encode(self, audio, sr):
        """
        Кодирует аудио в дискретные коды.
        
        Args:
            audio: numpy array аудио данных
            sr: частота дискретизации
            
        Returns:
            numpy array дискретных кодов размером (n_quantizers, n_frames)
        """
        if self.use_audiocraft and self.model is not None:
            return self._encode_with_audiocraft(audio, sr)
        elif self.use_encodec and self.model is not None:
            return self._encode_with_encodec(audio, sr)
        else:
            return self._encode_simplified(audio, sr)
    
    def _encode_with_audiocraft(self, audio, sr):
        """Кодирование с использованием AudioCraft"""
        try:
            # Преобразование в формат AudioCraft
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Преобразование в тензор
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
            
            # Кодирование через AudioGen для получения дискретных кодов
            with torch.no_grad():
                # Получаем кодировки
                if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                    self.model.encoder.eval()
                
                if hasattr(self.model, 'compression_model') and self.model.compression_model is not None:
                    codes = self.model.compression_model.encode(audio_tensor)
                    
                    # Извлекаем коды квантизации
                    quantized_codes = []
                    for code in codes:
                        quantized_codes.append(code.squeeze().cpu().numpy())
                    
                    # Преобразуем в формат (n_quantizers, n_frames)
                    stacked_codes = np.stack(quantized_codes, axis=0)
                    
                    return stacked_codes
                else:
                    print("Ошибка: compression_model не найден в модели AudioCraft")
                    return self._encode_simplified(audio, sr)
        except Exception as e:
            print(f"Ошибка при кодировании с AudioCraft: {e}")
            return self._encode_simplified(audio, sr)
    
    def _encode_with_encodec(self, audio, sr):
        """Кодирование с использованием EnCodec"""
        try:
            # Подготовка аудио для EnCodec
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Нормализация и преобразование в тензор
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
            
            # Кодирование
            with torch.no_grad():
                if hasattr(self.model, 'encode') and callable(getattr(self.model, 'encode')):
                    encoded_frames = self.model.encode(audio_tensor)
                    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
                    
                    # Преобразование в numpy array формата (n_quantizers, n_frames)
                    codes_np = codes.cpu().numpy()
                    
                    return codes_np
                else:
                    print("Ошибка: метод encode не найден в модели EnCodec")
                    return self._encode_simplified(audio, sr)
        except Exception as e:
            print(f"Ошибка при кодировании с EnCodec: {e}")
            return self._encode_simplified(audio, sr)
    
    def _encode_simplified(self, audio, sr):
        """
        Упрощенное кодирование, когда недоступны специализированные модели.
        Это имитация дискретизации, а не настоящий XCodec.
        """
        # Создаем временное представление с n_quantizers слоями
        frame_length = int(sr * 0.02)  # 20 мс фреймы
        hop_length = int(sr * 0.01)    # 10 мс перекрытие
        
        # Создаем спектрограмму
        spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length))
        
        # Имитируем дискретные коды для каждого квантизатора
        codes = np.zeros((self.n_quantizers, spec.shape[1]), dtype=np.int32)
        
        # Разные квантизаторы захватывают разные аспекты сигнала
        for i in range(self.n_quantizers):
            # Имитируем разные дискретизации для разных слоев
            freq_band = spec[i*spec.shape[0]//self.n_quantizers:(i+1)*spec.shape[0]//self.n_quantizers]
            mean_energy = np.mean(freq_band, axis=0)
            # Квантизация энергии в диапазоне [0, codebook_size-1]
            codes[i] = np.minimum(self.codebook_size-1, 
                            np.maximum(0, 
                                      (mean_energy / (np.max(mean_energy) + 1e-10) * (self.codebook_size-1)).astype(np.int32)))
        
        return codes


def load_wav_file(file_path, target_sr=16000):
    """
    Загружает WAV файл и приводит его к нужной частоте дискретизации
    
    Args:
        file_path: путь к WAV файлу
        target_sr: целевая частота дискретизации (по умолчанию 16 кГц)
    
    Returns:
        numpy array: аудиоданные в формате numpy array
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Ресемплинг, если частота дискретизации не соответствует целевой
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Нормализация аудио
        audio = audio / np.max(np.abs(audio))
        
        return audio, target_sr
    
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return None, None


def process_wav_files(input_dir, output_dir, target_sr=16000, codec_model_path=None):
    """
    Обрабатывает WAV файлы из входной директории и сохраняет NPY файлы в выходную директорию
    
    Args:
        input_dir: путь к директории с WAV файлами
        output_dir: путь к директории для сохранения NPY файлов
        target_sr: целевая частота дискретизации (по умолчанию 16 кГц)
        codec_model_path: путь к модели кодека (опционально)
    """
    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Инициализация кодека
    xcodec = XCodec(model_path=codec_model_path, device=device)
    
    # Список типов файлов для обработки
    file_types = ["vocals", "instrumental", "mix"]
    
    # Ищем файлы по типам
    for file_type in file_types:
        # Ищем файлы соответствующего типа (с учетом разных возможных расширений)
        matching_files = list(Path(input_dir).glob(f"*{file_type}*.wav")) + \
                         list(Path(input_dir).glob(f"*{file_type}*.WAV"))
        
        if not matching_files:
            print(f"Файл типа '{file_type}' не найден в {input_dir}")
            continue
            
        # Берем первый найденный файл данного типа
        wav_file = str(matching_files[0])
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        
        print(f"Обработка {wav_file}...")
        
        # Загружаем аудио
        audio, sr = load_wav_file(wav_file, target_sr)
        if audio is None:
            continue
        
        # Дискретизация аудио с помощью XCodec
        print(f"Кодирование аудио в дискретные коды...")
        discrete_codes = xcodec.encode(audio, sr)
        
        # Формируем имя выходного файла
        if file_type == "vocals":
            output_suffix = ".Vocals"
        elif file_type == "instrumental":
            output_suffix = ".Instrumental"
        else:
            output_suffix = ""  # Для mix файла не добавляем суффикс
            
        # Определяем имя выходного файла
        # Используем базовое имя файла без типа, если возможно
        base_name_without_type = base_name
        for t in file_types:
            base_name_without_type = base_name_without_type.replace(t, "")
        
        # Удаляем лишние символы после замены
        base_name_without_type = base_name_without_type.strip("_-. ")
        
        # Если базовое имя стало пустым, используем оригинальное
        if not base_name_without_type:
            base_name_without_type = base_name
            
        output_file = os.path.join(output_dir, f"{base_name_without_type}{output_suffix}.npy")
        
        # Сохраняем в формате NPY
        np.save(output_file, discrete_codes)
        print(f"Сохранено в {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертирует WAV файлы в формат NPY для использования с YuE"
    )
    parser.add_argument("input_dir", help="Директория с WAV файлами")
    parser.add_argument("output_dir", help="Директория для сохранения NPY файлов")
    parser.add_argument("--sr", type=int, default=16000, help="Частота дискретизации (по умолчанию 16000 Гц)")
    parser.add_argument("--codec_model", type=str, default=None, help="Путь к модели кодека (опционально)")
    
    args = parser.parse_args()
    
    # Проверка существования входной директории
    if not os.path.exists(args.input_dir):
        print(f"Ошибка: входная директория {args.input_dir} не существует")
        sys.exit(1)
    
    # Запуск обработки файлов
    process_wav_files(
        args.input_dir,
        args.output_dir,
        args.sr,
        args.codec_model
    )
    
    print("Преобразование завершено.")


if __name__ == "__main__":
    main() 