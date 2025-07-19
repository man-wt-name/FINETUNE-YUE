# FINETUNE-YUE: Тонкая Настройка Модели YuE для Генерации Аудио

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Введение

FINETUNE-YUE — это проект для тонкой настройки (fine-tuning) мультимодальной модели генерации аудио YuE с использованием LoRA (Low-Rank Adaptation), DeepSpeed для распределенного обучения и пользовательских инструментов для обработки аудио-данных. Проект поддерживает предобработку аудио в форматы, подходящие для обучения (например, XCodec, npy), токенизацию, подсчет токенов и интерфейс Gradio для инференса.

Ключевые особенности:
- Поддержка LoRA для эффективного fine-tuning.
- Интеграция с Hugging Face Transformers и PEFT.
- Обработка аудио-датасетов (вокал, инструменталы, миксы).
- Оптимизировано для Kaggle (Jupyter Notebook, Python 3.11, CUDA 12.6).
- Логирование с WandB (опционально).

Проект следует лучшим практикам Python: PEP8, type hints, docstrings, разделение зависимостей.

## Требования

- ОС: Ubuntu 22.04 (или совместимая, как в Kaggle).
- Python: 3.11.
- GPU: Поддержка CUDA 12.6 (в Kaggle — 2x T4 или P100).
- Корневой каталог в Kaggle: `/kaggle/working/FINETUNE-YUE`.

## Инструкция по Установке

1. **Клонируйте репозиторий**:
   ```
   git clone https://github.com/your-repo/FINETUNE-YUE.git
   cd FINETUNE-YUE
   ```

2. **Установите зависимости**:
   - Основные (production):
     ```
     pip install -r requirements.txt
     ```
     (Включает torch==2.4.1, torchaudio==2.4.1, transformers==4.41.2 и т.д.)

   - Для разработки (dev, опционально, включает wandb и nltk):
     ```
     pip install -r requirements-dev.txt
     ```

3. **Внешние зависимости** (устанавливаются вручную):
   - **xcodec_mini_infer и SoundStream**: Скачайте и установите из внешних источников (см. их репозитории). Разместите в `xcodec_mini_infer/` (config.yaml и ckpt.pth).
   - **ffmpeg**: Установите системно:
     ```
     sudo apt-get update && sudo apt-get install ffmpeg
     ```
   - **C++ расширения**: Скомпилируйте helpers.cpp в `core/datasets/` с помощью Makefile:
     ```
     cd core/datasets && make
     ```

4. **Настройка окружения в Kaggle**:
   - Загрузите датасет в `/kaggle/input/`.
   - Установите зависимости в Jupyter Notebook:
     ```
     !pip install -r /kaggle/working/FINETUNE-YUE/requirements.txt
     ```

5. **Проверка установки**:
   - Запустите `python -c "import torch; print(torch.__version__)"` — должно вывести 2.4.1.
   - Если используете WandB: Установите API ключ в переменных окружения: `export WANDB_API_KEY=your_key`.

## Инструкция по Созданию Конфигураций

Конфигурации хранятся в `config/` и `example/`. Они в формате JSON/YAML.

1. **DeepSpeed конфиг (config/ds_config_zero2.json)**:
   - Это шаблон для распределенного обучения (Zero-2).
   - Ключевые параметры:
     - `"zero_optimization"`: Настройки для оптимизации памяти (CPU offload и т.д.).
     - `"fp16"`: Включите для FP16.
   - Пример редактирования: Откройте в редакторе и измените `"train_batch_size"` на ваш global_batch_size.
   - Создание: Скопируйте шаблон и настройте под GPU (например, для Kaggle с 2 GPU: batch_size=2).

2. **Смесь данных (example/dummy_data_mixture_cfg.yml)**:
   - Для парсинга датасета в `core/parse_mixture.py`.
   - Ключевые параметры:
     - `TOKEN_COUNT_LOG_DIR`: Директория логов подсчета токенов.
     - `GLOBAL_BATCH_SIZE`: Размер батча (например, 64).
     - `SEQ_LEN`: Длина последовательности (например, 8192).
     - `1_ROUND`: Количество раундов повторения датасета (например, 1).
   - Создание: Скопируйте из example и настройте. Пример:
     ```
     TOKEN_COUNT_LOG_DIR: ./count_token_logs/
     GLOBAL_BATCH_SIZE: 64
     SEQ_LEN: 8192
     1_ROUND: 1
     ```

3. **Другие конфиги**:
   - Для XCodec: `xcodec_mini_infer/final_ckpt/config.yaml` — настройте bandwidth, sample_rate.
   - Создайте новые YAML/JSON по шаблонам в `example/`.

## Инструкции по Работе

### Общие Рекомендации
- Все скрипты запускаются из корня проекта: `python scripts/script_name.py --args`.
- Пути адаптированы для Kaggle (см. adjust_paths_for_kaggle в infer.py).
- Быстрый запуск: Малый batch, мало итераций (для теста на CPU/GPU).
- Оптимальный: Сбалансированный batch, bf16, multi-GPU для скорости/качества.

#### 1. tools/audio_to_npy.py (Конвертация аудио в .npy)
   - **Описание**: Конвертирует аудио (wav) в npy с использованием XCodec.
   - **Аргументы**:
     - `--input_dir` (str, required, default='/kaggle/input/data'): Директория с vocal.wav, instrumental.wav, mix.wav.
     - `--output_dir` (str, required, default='/kaggle/working/FINETUNE-YUE/npy'): Выходная директория для .npy.
     - `--config_path` (str, default='xcodec_mini_infer/final_ckpt/config.yaml'): Путь к XCodec config.
     - `--ckpt_path` (str, default='xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'): Путь к чекпоинту XCodec.
     - `--target_bw` (float, default=6.0): Целевая полоса пропускания.
     - `--sample_rate` (int, default=16000): Частота дискретизации.
   - **Пример быстрый**: `python tools/audio_to_npy.py --input_dir example/ --output_dir output/npy --target_bw 3.0`
   - **Пример оптимальный**: `python tools/audio_to_npy.py --input_dir /kaggle/input/my-audio --output_dir /kaggle/working/FINETUNE-YUE/npy --target_bw 6.0 --sample_rate 16000`

#### 2. scripts/preprocess_data.py (Предобработка данных)
   - **Описание**: Обрабатывает jsonl и npy в mmap для обучения.
   - **Аргументы** (основные из core/preprocess_data_conditional_xcodec_segment.py):
     - `--data_setting` (str, default='dummy'): Настройка данных.
     - `--mode_type` (str, choices=['cot', 'icl_cot'], default='cot'): Режим.
     - `--tokenizer_model` (str, default='mm_tokenizer_v0.2_hf/tokenizer.model'): Путь к токенизатору.
     - `--audio_prompt_modes` (list, default=['dual', 'inst', 'vocal', 'mixture']): Режимы промптов.
     - `--output_prefix` (str, default='/kaggle/working/FINETUNE-YUE/output/processed'): Префикс вывода.
   - **Пример быстрый**: `python scripts/preprocess_data.py dummy cot mm_tokenizer_v0.2_hf/tokenizer.model --audio_prompt_modes dual`
   - **Пример оптимальный**: `python scripts/preprocess_data.py my_data icl_cot mm_tokenizer_v0.2_hf/tokenizer.model --audio_prompt_modes dual inst vocal mixture --output_prefix /kaggle/working/FINETUNE-YUE/processed`

#### 3. scripts/count_tokens.py (Подсчет токенов)
   - **Описание**: Подсчитывает токены в mmap.
   - **Аргументы**:
     - `--parent_dir` (str, required): Директория с mmap.
     - `--log_dir` (str, default='./count_token_logs/'): Директория логов.
   - **Пример быстрый**: `python scripts/count_tokens.py --parent_dir output/mmap --log_dir logs/`
   - **Пример оптимальный**: `python scripts/count_tokens.py --parent_dir /kaggle/working/FINETUNE-YUE/processed --log_dir /kaggle/working/FINETUNE-YUE/logs`

#### 4. core/parse_mixture.py (Парсинг смеси данных)
   - **Описание**: Генерирует DATA_PATH и TRAIN_ITERS из YAML.
   - **Аргументы**:
     - `-c` / `--config` (str, required): Путь к YAML (см. выше).
   - **Пример быстрый**: `python core/parse_mixture.py -c example/dummy_data_mixture_cfg.yml`
   - **Пример оптимальный**: `python core/parse_mixture.py -c my_config.yml` (с SEQ_LEN=8192, GLOBAL_BATCH_SIZE=64)

#### 5. scripts/run_finetune.py (Запуск fine-tuning)
   - **Описание**: Обертка для train_lora.py с env vars.
   - **Аргументы** (через env или код; см. get_args()):
     - `--help`: Показать помощь.
     - Env: NUM_GPUS (int, default=8), GLOBAL_BATCH_SIZE (int), SEQ_LENGTH (int, default=8192), etc. (см. код).
   - **Пример быстрый**: `NUM_GPUS=1 GLOBAL_BATCH_SIZE=1 SEQ_LENGTH=1024 python scripts/run_finetune.py`
   - **Пример оптимальный**: `NUM_GPUS=2 GLOBAL_BATCH_SIZE=64 SEQ_LENGTH=8192 USE_BF16=True python scripts/run_finetune.py`

#### 6. infer.py (Инференс)
   - **Описание**: Полный пайплайн: от аудио до fine-tuning и вывода.
   - **Аргументы** (много, см. get_args()):
     - `--input_dir` (str, required): Входные данные.
     - `--output_dir` (str, default='./output'): Выход.
     - `--data_setting` (str, default='dummy'): Настройка.
     - `--mode_type` (str, default='cot'): Режим.
     - И многие другие (см. parser.add_argument).
   - **Пример быстрый**: `python infer.py --input_dir example/ --data_setting dummy --num_gpus 1 --train_iters 10`
   - **Пример оптимальный**: `python infer.py --input_dir /kaggle/input/my-data --data_setting my_data --num_gpus 2 --num_train_epochs 10 --use_bf16 --use_wandb`

#### 7. gradio_interface.py (Gradio UI)
   - **Описание**: Интерфейс для инференса.
   - **Аргументы**: Нет прямых; запускается как `python gradio_interface.py`.
   - **Пример**: Просто запустите для UI.

#### 8. tools/codecmanipulator.py (Манипуляция кодеком)
   - **Описание**: Инструмент для манипуляции кодеком (например, сегментация).
   - **Аргументы**: (Проверьте код; похожи на preprocess).
   - **Пример**: `python tools/codecmanipulator.py --input npy/ --output processed/`

#### 9. tools/count_mmap_token.py (Подсчет токенов в mmap)
   - **Описание**: Альтернативный подсчет.
   - **Аргументы**: `--file` (str, required): Путь к mmap.
   - **Пример**: `python tools/count_mmap_token.py --file processed/data.mmap`

#### 10. scripts/train_lora.py (Основной скрипт LoRA обучения)
    - **Описание**: Непосредственно обучает LoRA (вызывается из run_finetune.py).
    - **Аргументы** (из core/arguments.py): `--model-name-or-path`, `--output-dir`, `--lora-r` и т.д. (см. выше).
    - **Пример**: Запускается через run_finetune.py.

## Подробная Информация по Созданию Датасета

Датасет — ключевой для fine-tuning. Проект ожидает аудио-треки (вокал, инструменталы, миксы) в wav, конвертированные в npy и jsonl.

1. **Типы треков**:
   - **Вокальные**: Чистый вокал (vocal.wav) — для обучения на голосе. Рекомендуется: 40-60% датасета.
   - **Инструментальные**: Инструменталы (instrumental.wav) — для фонов. 20-30%.
   - **Миксы**: Полные треки (mix.wav) — для генерации комбинированного аудио. 20-30%.
   - Баланс: Для оптимального качества — 50% вокал, 30% инструменталы, 20% миксы. Используйте разнообразные жанры (поп, рок) для генерализации.

2. **Количество треков**:
   - Минимально: 10-50 для теста (dummy в example: 1-2 трека, ~1-10 MB).
   - Оптимально: 1000+ треков для хорошего обучения (общий размер ~10-100 GB, ~1-5M токенов).
   - Эффективно: 5000+ треков для production (с SEQ_LEN=8192, ~1M токенов на трек ~30 сек, общий ~5B токенов).
   - Длина: Треки 10-60 сек, sample_rate=16000. Обрезайте/сегментируйте длинные с codecmanipulator.py.

3. **Шаги создания**:
   - **Сбор данных**: Соберите wav в директорию (например, из открытых датасетов как AudioSet или custom).
   - **Конвертация**: `python tools/audio_to_npy.py --input_dir your_audio/ --output_dir npy/` (генерирует dummy.npy, dummy.Vocals.npy и т.д.).
   - **Создание jsonl**: См. example/jsonl/dummy.msa.xcodec_16k.jsonl. Формат: JSONL с метаданными (пути npy, текстовые промпты, типы: "vocal", "instrumental").
     - Пример строки: `{"audio_path": "npy/dummy.npy", "prompt": "Generate vocal mix", "type": "mixture"}`
   - **Предобработка**: `python scripts/preprocess_data.py ...` (создает mmap в output_prefix, токенизирует с mmtokenizer.py).
   - **Подсчет токенов**: `python scripts/count_tokens.py ...` (логи в log_dir для parse_mixture).
   - **Парсинг**: `python core/parse_mixture.py -c your_cfg.yml` (выводит DATA_PATH и TRAIN_ITERS для обучения).
   - **Проверка**: Используйте count_mmap_token.py для верификации токенов.

## Подробная Инструкция по Тренировке LoRA

### Разбор Параметров Обучения (из core/arguments.py, scripts/run_finetune.py и train_lora.py)
- **LoRA-специфичные**:
  - `--lora-r` (int, default=64): Rank LoRA (меньше — быстрее, но хуже адаптация; оптимально 32-128).
  - `--lora-alpha` (int, default=32): Scaling factor (больше — сильнее адаптация, но риск нестабильности; 16-64).
  - `--lora-dropout` (float, default=0.1): Dropout для LoRA (0.05-0.2 для регуляризации).
  - `--lora-target-modules` (str, default='q_proj k_proj v_proj o_proj'): Модули transformer для адаптации (добавьте gate_proj для большего охвата).

- **Общие**:
  - `--seq-length` (int, default=8192): Длина последовательности (больше — лучше контекст, но больше памяти; 4096-16384).
  - `--global-batch-size` (int): Размер батча (больше — быстрее сходимость, но требует GPU; 32-128).
  - `--train-iters` (int, default=150): Итерации (из parse_mixture; зависит от датасета, 100-1000).
  - `--num-train-epochs` (int, default=10): Эпохи (5-20 для баланса).
  - `--use-bf16` (bool): BF16 для скорости и памяти (рекомендовано для CUDA 12+; альтернатива fp16).
  - `--deepspeed` (str, default='config/ds_config_zero2.json'): Конфиг для оптимизации (Zero-2 для multi-GPU).
  - `--logging-steps` (int, default=5): Интервал логирования (1-10).
  - `--save-steps` (int, default=5): Интервал сохранений (5-20).
  - `--use-wandb` (bool): Логи в WandB (включите для мониторинга).
  - `--model-name-or-path` (str, default='m-a-p/YuE-s1-7B-anneal-en-cot'): Базовая модель (HF hub).
  - `--data-path` (str): Из parse_mixture.
  - `--data-cache-path` (str, default='/kaggle/working/FINETUNE-YUE/cache'): Кэш данных.
  - `--data-split` (str, default='900,50,50'): Разделение train/val/test.

### Рекомендации по Режимам Обучения
- **Быстрое обучение** (для теста, ~1-2 часа на 1 GPU):
  - Параметры: `--global-batch-size 1 --train-iters 50 --seq-length 1024 --lora-r 16 --use-bf16 --num-train-epochs 1`.
  - Когда: Для быстрого прототипа на малом датасете (10 треков), проверки setup.
  - Минусы: Низкое качество, риск недообучения.

- **Оптимальное обучение** (соотношение скорости/качества, ~1-3 дня на 2 GPU):
  - Параметры: `--global-batch-size 64 --train-iters 150 --seq-length 8192 --lora-r 64 --use-bf16 --num-train-epochs 10 --use-wandb --deepspeed config/ds_config_zero2.json`.
  - Когда: Для 1000+ треков, баланс (Deepspeed для памяти, bf16 для скорости).
  - Плюсы: Хорошее качество без перерасхода ресурсов; мониторьте loss в WandB.

- **Самый эффективный режим** (максимальное качество, ~1 неделя на multi-GPU):
  - Параметры: `--global-batch-size 128 --train-iters 500 --seq-length 16384 --lora-r 128 --use-bf16 --num-train-epochs 20 --deepspeed config/ds_config_zero2.json --use-wandb --data-split '900,50,50' --lora-alpha 64 --lora-dropout 0.05`.
  - Когда: Для 5000+ треков, полный датасет (баланс вокал/инструменталы/миксы).
  - Плюсы: Лучшая генерализация, высокое качество генерации; используйте multi-GPU (NUM_GPUS=8) и DeepSpeed для масштаба.
  - Советы: Мониторьте с WandB (loss, ppl); добавьте early stopping если loss не падает; используйте большой датасет для избежания переобучения; тестовый запуск на val-set после каждой эпохи.

### Полный Пайплайн Обучения
1. Подготовьте датасет (см. выше).
2. Парсинг: Получите DATA_PATH и TRAIN_ITERS.
3. Запустите: `python scripts/run_finetune.py` с env vars.
4. Мониторинг: Логи в output_dir, WandB.
5. После: Используйте infer.py или Gradio для теста генерации.

Если проблемы — проверьте логи в output/. Для кастомизации обратитесь к core/arguments.py. 

## Troubleshooting
- **Path not found in Kaggle**: Убедитесь, что датасет загружен в /kaggle/input/. Проверьте basename в adjust_paths.
- **Dependency errors**: Проверьте версии в requirements.txt.
- **Subprocess failed**: Смотрите stderr в логах. 