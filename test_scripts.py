#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для тестирования всех основных функций из папки scripts проекта YuE finetuning.
Выполняет проверку работоспособности и генерирует подробный отчет.
"""

import os
import sys
import time
import argparse
import subprocess
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime


class TestResult:
    """Класс для хранения результатов тестирования"""
    def __init__(self, script_name, status, message, execution_time=0, details=None):
        self.script_name = script_name
        self.status = status  # "SUCCESS", "FAILED", "SKIPPED"
        self.message = message
        self.execution_time = execution_time
        self.details = details or {}
    
    def __str__(self):
        return f"{self.script_name}: {self.status} - {self.message} ({self.execution_time:.2f}s)"


class ScriptTester:
    """Класс для тестирования скриптов"""
    
    def __init__(self, scripts_dir="scripts", output_dir="test_results", verbose=True):
        self.scripts_dir = Path(scripts_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results = []
        
        # Создаем директорию для результатов, если она не существует
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Создаем временную директорию для тестовых данных
        self.test_data_dir = self.output_dir / "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def log(self, message):
        """Выводит сообщение, если включен режим verbose"""
        if self.verbose:
            print(message)
    
    def find_scripts(self):
        """Находит все Python скрипты в директории scripts"""
        python_scripts = list(self.scripts_dir.glob("*.py"))
        self.log(f"Найдено {len(python_scripts)} Python скриптов в директории {self.scripts_dir}")
        return python_scripts
    
    def import_script(self, script_path):
        """Импортирует скрипт как модуль"""
        try:
            module_name = script_path.stem
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.log(f"Ошибка при импорте скрипта {script_path}: {e}")
            return None
    
    def test_script_import(self, script_path):
        """Тестирует импорт скрипта"""
        start_time = time.time()
        try:
            module = self.import_script(script_path)
            if module:
                execution_time = time.time() - start_time
                return TestResult(
                    script_path.name,
                    "SUCCESS",
                    "Скрипт успешно импортирован",
                    execution_time,
                    {"module": module.__name__}
                )
            else:
                execution_time = time.time() - start_time
                return TestResult(
                    script_path.name,
                    "FAILED",
                    "Ошибка при импорте скрипта",
                    execution_time
                )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                script_path.name,
                "FAILED",
                f"Исключение при импорте: {str(e)}",
                execution_time,
                {"exception": traceback.format_exc()}
            )
    
    def test_convert_py(self):
        """Тестирует скрипт convert.py"""
        script_name = "convert.py"
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, "SKIPPED", "Скрипт не найден")
        
        self.log(f"\nТестирование {script_name}...")
        start_time = time.time()
        
        # Создаем тестовые данные (заглушки WAV файлов)
        test_input_dir = self.test_data_dir / "wav_input"
        test_output_dir = self.test_data_dir / "npy_output"
        os.makedirs(test_input_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Создаем заглушки WAV файлов для тестирования
        try:
            # Проверяем наличие numpy и scipy
            import numpy as np
            from scipy.io import wavfile
            
            # Создаем тестовые WAV файлы
            sample_rate = 16000
            duration = 1  # 1 секунда
            t = np.linspace(0, duration, sample_rate * duration)
            
            # Vocals файл (синусоида 440 Гц)
            vocals_data = np.sin(2 * np.pi * 440 * t)
            wavfile.write(test_input_dir / "test_vocals.wav", sample_rate, vocals_data.astype(np.float32))
            
            # Instrumental файл (синусоида 880 Гц)
            instrumental_data = np.sin(2 * np.pi * 880 * t)
            wavfile.write(test_input_dir / "test_instrumental.wav", sample_rate, instrumental_data.astype(np.float32))
            
            # Mix файл (сумма vocals и instrumental)
            mix_data = vocals_data + instrumental_data
            wavfile.write(test_input_dir / "test_mix.wav", sample_rate, mix_data.astype(np.float32))
            
            self.log("Созданы тестовые WAV файлы")
        except ImportError as e:
            return TestResult(
                script_name,
                "SKIPPED",
                f"Не удалось создать тестовые WAV файлы: {e}",
                time.time() - start_time
            )
        
        # Запускаем скрипт convert.py
        try:
            cmd = [
                sys.executable,
                str(script_path),
                str(test_input_dir),
                str(test_output_dir),
                "--sr", "16000"
            ]
            
            self.log(f"Выполнение команды: {' '.join(cmd)}")
            
            # Запускаем процесс с перехватом вывода
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60  # Ограничиваем время выполнения 60 секундами
            )
            
            output = process.stdout
            error = process.stderr
            
            # Проверяем результат выполнения
            if process.returncode == 0:
                # Проверяем, были ли созданы NPY файлы
                npy_files = list(test_output_dir.glob("*.npy"))
                if npy_files:
                    return TestResult(
                        script_name,
                        "SUCCESS",
                        f"Скрипт успешно выполнен. Создано {len(npy_files)} NPY файлов.",
                        time.time() - start_time,
                        {"output": output, "error": error, "files": [str(f) for f in npy_files]}
                    )
                else:
                    return TestResult(
                        script_name,
                        "FAILED",
                        "Скрипт выполнен, но NPY файлы не созданы",
                        time.time() - start_time,
                        {"output": output, "error": error}
                    )
            else:
                return TestResult(
                    script_name,
                    "FAILED",
                    f"Скрипт завершился с ошибкой (код {process.returncode})",
                    time.time() - start_time,
                    {"output": output, "error": error}
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                script_name,
                "FAILED",
                "Превышено время выполнения скрипта (60 секунд)",
                time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                script_name,
                "FAILED",
                f"Исключение при выполнении: {str(e)}",
                time.time() - start_time,
                {"exception": traceback.format_exc()}
            )
    
    def test_count_tokens_py(self):
        """Тестирует скрипт count_tokens.py"""
        script_name = "count_tokens.py"
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, "SKIPPED", "Скрипт не найден")
        
        self.log(f"\nТестирование {script_name}...")
        start_time = time.time()
        
        # Создаем тестовую директорию с .bin файлами
        test_bin_dir = self.test_data_dir / "bin_files"
        os.makedirs(test_bin_dir, exist_ok=True)
        
        # Создаем заглушку .bin файла
        try:
            with open(test_bin_dir / "test.bin", "wb") as f:
                f.write(b"TEST_BINARY_DATA" * 100)
            
            self.log("Создан тестовый .bin файл")
        except Exception as e:
            return TestResult(
                script_name,
                "SKIPPED",
                f"Не удалось создать тестовый .bin файл: {e}",
                time.time() - start_time
            )
        
        # Запускаем скрипт count_tokens.py
        try:
            cmd = [
                sys.executable,
                str(script_path),
                str(test_bin_dir)
            ]
            
            self.log(f"Выполнение команды: {' '.join(cmd)}")
            
            # Запускаем процесс с перехватом вывода
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30  # Ограничиваем время выполнения 30 секундами
            )
            
            output = process.stdout
            error = process.stderr
            
            # Проверяем результат выполнения
            if process.returncode == 0:
                # Проверяем, была ли создана директория для логов
                log_dir = Path("./count_token_logs")
                if log_dir.exists() and log_dir.is_dir():
                    log_files = list(log_dir.glob("*.log"))
                    return TestResult(
                        script_name,
                        "SUCCESS",
                        f"Скрипт успешно выполнен. Создано {len(log_files)} лог-файлов.",
                        time.time() - start_time,
                        {"output": output, "error": error, "log_files": [str(f) for f in log_files]}
                    )
                else:
                    return TestResult(
                        script_name,
                        "PARTIAL",
                        "Скрипт выполнен, но директория с логами не найдена",
                        time.time() - start_time,
                        {"output": output, "error": error}
                    )
            else:
                return TestResult(
                    script_name,
                    "FAILED",
                    f"Скрипт завершился с ошибкой (код {process.returncode})",
                    time.time() - start_time,
                    {"output": output, "error": error}
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                script_name,
                "FAILED",
                "Превышено время выполнения скрипта (30 секунд)",
                time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                script_name,
                "FAILED",
                f"Исключение при выполнении: {str(e)}",
                time.time() - start_time,
                {"exception": traceback.format_exc()}
            )
    
    def test_preprocess_data_py(self):
        """Тестирует скрипт preprocess_data.py"""
        script_name = "preprocess_data.py"
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, "SKIPPED", "Скрипт не найден")
        
        self.log(f"\nТестирование {script_name}...")
        start_time = time.time()
        
        # Проверяем наличие примера данных
        example_dir = Path("example")
        if not example_dir.exists() or not (example_dir / "jsonl" / "dummy.msa.xcodec_16k.jsonl").exists():
            return TestResult(
                script_name,
                "SKIPPED",
                "Не найдены необходимые тестовые данные в директории example",
                time.time() - start_time
            )
        
        # Проверяем наличие токенизатора
        tokenizer_path = Path("mm_tokenizer_v0.2_hf/tokenizer.model")
        if not tokenizer_path.exists():
            return TestResult(
                script_name,
                "SKIPPED",
                "Не найден файл токенизатора",
                time.time() - start_time
            )
        
        # Запускаем скрипт preprocess_data.py в режиме cot
        try:
            cmd = [
                sys.executable,
                str(script_path),
                "dummy",
                "cot",
                str(tokenizer_path)
            ]
            
            self.log(f"Выполнение команды: {' '.join(cmd)}")
            
            # Запускаем процесс с перехватом вывода
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120  # Ограничиваем время выполнения 2 минутами
            )
            
            output = process.stdout
            error = process.stderr
            
            # Проверяем результат выполнения
            if process.returncode == 0:
                # Проверяем, были ли созданы выходные файлы
                mmap_dir = example_dir / "mmap"
                if mmap_dir.exists() and mmap_dir.is_dir():
                    return TestResult(
                        script_name,
                        "SUCCESS",
                        "Скрипт успешно выполнен в режиме cot",
                        time.time() - start_time,
                        {"output": output, "error": error}
                    )
                else:
                    return TestResult(
                        script_name,
                        "PARTIAL",
                        "Скрипт выполнен, но выходные файлы не найдены",
                        time.time() - start_time,
                        {"output": output, "error": error}
                    )
            else:
                return TestResult(
                    script_name,
                    "FAILED",
                    f"Скрипт завершился с ошибкой (код {process.returncode})",
                    time.time() - start_time,
                    {"output": output, "error": error}
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                script_name,
                "FAILED",
                "Превышено время выполнения скрипта (120 секунд)",
                time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                script_name,
                "FAILED",
                f"Исключение при выполнении: {str(e)}",
                time.time() - start_time,
                {"exception": traceback.format_exc()}
            )
    
    def test_run_finetune_py(self):
        """Тестирует скрипт run_finetune.py (только проверка аргументов)"""
        script_name = "run_finetune.py"
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, "SKIPPED", "Скрипт не найден")
        
        self.log(f"\nТестирование {script_name}...")
        start_time = time.time()
        
        # Запускаем скрипт с аргументом --help для проверки
        try:
            cmd = [
                sys.executable,
                str(script_path),
                "--help"
            ]
            
            self.log(f"Выполнение команды: {' '.join(cmd)}")
            
            # Запускаем процесс с перехватом вывода
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            
            output = process.stdout
            error = process.stderr
            
            # Проверяем результат выполнения
            if "YuE Fine-tuning Script Help" in output:
                return TestResult(
                    script_name,
                    "SUCCESS",
                    "Скрипт успешно отобразил справку",
                    time.time() - start_time,
                    {"output": output, "error": error}
                )
            else:
                return TestResult(
                    script_name,
                    "PARTIAL",
                    "Скрипт выполнен, но справка не содержит ожидаемого текста",
                    time.time() - start_time,
                    {"output": output, "error": error}
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                script_name,
                "FAILED",
                "Превышено время выполнения скрипта (30 секунд)",
                time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                script_name,
                "FAILED",
                f"Исключение при выполнении: {str(e)}",
                time.time() - start_time,
                {"exception": traceback.format_exc()}
            )
    
    def test_train_lora_py(self):
        """Тестирует скрипт train_lora.py (только импорт и проверка аргументов)"""
        script_name = "train_lora.py"
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return TestResult(script_name, "SKIPPED", "Скрипт не найден")
        
        self.log(f"\nТестирование {script_name}...")
        start_time = time.time()
        
        # Проверяем импорт скрипта
        import_result = self.test_script_import(script_path)
        if import_result.status != "SUCCESS":
            return import_result
        
        # Проверяем наличие основных функций в скрипте
        try:
            module = import_result.details.get("module")
            if not module:
                return TestResult(
                    script_name,
                    "FAILED",
                    "Не удалось получить модуль скрипта",
                    time.time() - start_time
                )
            
            module_obj = sys.modules.get(module)
            if not module_obj:
                return TestResult(
                    script_name,
                    "FAILED",
                    "Не удалось получить объект модуля",
                    time.time() - start_time
                )
            
            # Проверяем наличие основных функций
            required_functions = ["main", "build_train_valid_test_datasets", "create_and_configure_model"]
            missing_functions = [f for f in required_functions if not hasattr(module_obj, f)]
            
            if missing_functions:
                return TestResult(
                    script_name,
                    "PARTIAL",
                    f"В скрипте отсутствуют функции: {', '.join(missing_functions)}",
                    time.time() - start_time
                )
            
            return TestResult(
                script_name,
                "SUCCESS",
                "Скрипт успешно импортирован и содержит все необходимые функции",
                time.time() - start_time,
                {"functions": required_functions}
            )
        except Exception as e:
            return TestResult(
                script_name,
                "FAILED",
                f"Исключение при проверке функций: {str(e)}",
                time.time() - start_time,
                {"exception": traceback.format_exc()}
            )
    
    def run_all_tests(self):
        """Запускает все тесты для скриптов"""
        self.log("\n" + "=" * 60)
        self.log("ЗАПУСК ТЕСТИРОВАНИЯ СКРИПТОВ YUE FINETUNING")
        self.log("=" * 60)
        
        # Тестируем основные скрипты
        self.results.append(self.test_convert_py())
        self.results.append(self.test_count_tokens_py())
        self.results.append(self.test_preprocess_data_py())
        self.results.append(self.test_run_finetune_py())
        self.results.append(self.test_train_lora_py())
        
        # Тестируем остальные скрипты на импорт
        all_scripts = self.find_scripts()
        tested_scripts = {
            "convert.py", "count_tokens.py", "preprocess_data.py",
            "run_finetune.py", "train_lora.py"
        }
        
        for script_path in all_scripts:
            if script_path.name not in tested_scripts:
                self.log(f"\nТестирование импорта {script_path.name}...")
                self.results.append(self.test_script_import(script_path))
        
        return self.results
    
    def generate_report(self):
        """Генерирует отчет о результатах тестирования"""
        if not self.results:
            self.log("Нет результатов тестирования. Запустите run_all_tests() сначала.")
            return
        
        # Подсчет статистики
        total = len(self.results)
        success = len([r for r in self.results if r.status == "SUCCESS"])
        failed = len([r for r in self.results if r.status == "FAILED"])
        partial = len([r for r in self.results if r.status == "PARTIAL"])
        skipped = len([r for r in self.results if r.status == "SKIPPED"])
        
        # Формируем отчет
        report = []
        report.append("=" * 80)
        report.append(f"ОТЧЕТ О ТЕСТИРОВАНИИ СКРИПТОВ YUE FINETUNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append(f"Всего скриптов: {total}")
        report.append(f"Успешно: {success}")
        report.append(f"Частично успешно: {partial}")
        report.append(f"Не пройдено: {failed}")
        report.append(f"Пропущено: {skipped}")
        report.append("=" * 80)
        report.append("ДЕТАЛИ ТЕСТИРОВАНИЯ:")
        report.append("=" * 80)
        
        # Добавляем детали по каждому скрипту
        for result in self.results:
            report.append(f"\n{result.script_name} - {result.status}")
            report.append("-" * 40)
            report.append(f"Сообщение: {result.message}")
            report.append(f"Время выполнения: {result.execution_time:.2f} секунд")
            
            # Добавляем детали, если они есть
            if result.details:
                if "output" in result.details and result.details["output"]:
                    report.append("\nВывод:")
                    output_lines = result.details["output"].split("\n")
                    # Ограничиваем вывод 10 строками
                    if len(output_lines) > 10:
                        output_lines = output_lines[:5] + ["..."] + output_lines[-5:]
                    report.append("\n".join(output_lines))
                
                if "error" in result.details and result.details["error"]:
                    report.append("\nОшибки:")
                    error_lines = result.details["error"].split("\n")
                    # Ограничиваем вывод ошибок 10 строками
                    if len(error_lines) > 10:
                        error_lines = error_lines[:5] + ["..."] + error_lines[-5:]
                    report.append("\n".join(error_lines))
                
                if "exception" in result.details:
                    report.append("\nИсключение:")
                    exception_lines = result.details["exception"].split("\n")
                    # Ограничиваем вывод исключения 10 строками
                    if len(exception_lines) > 10:
                        exception_lines = exception_lines[:5] + ["..."] + exception_lines[-5:]
                    report.append("\n".join(exception_lines))
            
            report.append("-" * 40)
        
        # Сохраняем отчет в файл
        report_text = "\n".join(report)
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        self.log(f"\nОтчет сохранен в файл: {report_file}")
        
        # Выводим краткую статистику
        self.log("\n" + "=" * 60)
        self.log("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        self.log(f"Всего скриптов: {total}")
        self.log(f"Успешно: {success}")
        self.log(f"Частично успешно: {partial}")
        self.log(f"Не пройдено: {failed}")
        self.log(f"Пропущено: {skipped}")
        self.log("=" * 60)
        
        return report_file


def main():
    """Основная функция для запуска тестирования скриптов"""
    parser = argparse.ArgumentParser(
        description="Тестирование скриптов YuE finetuning"
    )
    parser.add_argument("--scripts-dir", default="scripts",
                      help="Директория со скриптами (по умолчанию: scripts)")
    parser.add_argument("--output-dir", default="test_results",
                      help="Директория для результатов тестирования (по умолчанию: test_results)")
    parser.add_argument("--quiet", action="store_true",
                      help="Не выводить подробные сообщения")
    
    args = parser.parse_args()
    
    # Создаем тестер и запускаем тесты
    tester = ScriptTester(
        scripts_dir=args.scripts_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    tester.run_all_tests()
    tester.generate_report()


if __name__ == "__main__":
    main() 