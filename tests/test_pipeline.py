import pytest
from unittest import mock
import infer  # или другие модули

@pytest.fixture
def mock_args():
    return mock.Mock(input_dir='test_input', output_dir='test_output')

def test_full_pipeline(mock_args):
    with mock.patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        infer.run_pipeline(mock_args)
        assert mock_run.called  # Проверяем вызовы

// Добавляю тесты для отдельных шагов: test_audio_conversion, etc. 