import os
from typing import Dict, Any


class Config:
    """
    Центральный класс конфигурации для системы обработки новостей части B.
    Содержит пути к файлам, настройки моделей и параметры обработки.
    """

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATHS = {
        'raw': os.path.join(BASE_PATH, '..', 'data', 'raw'),
        'processed': os.path.join(BASE_PATH, '..', 'data', 'processed'),
        'results': os.path.join(BASE_PATH, '..', 'data', 'results'),
        'logs': os.path.join(BASE_PATH, '..', 'logs')
    }

    DATA_FILES = {
        'candles': 'train_candles.csv',
        'news': 'train_news.csv',
        'processed_sentiment': 'processed_sentiment_features.csv',
        'patchtst_sequences': 'patchtst_sequences.npz',
        'patchtst_config': 'patchtst_config.json'
    }

    FINBERT_CONFIG = {
        'model_name': 'ProsusAI/finbert',
        'max_length': 512,
        'batch_size': 16,
        'device': 'cpu'
    }

    # Настройки PatchTST
    PATCHTST_CONFIG = {
        'context_length': 20,
        'prediction_length': 1,
        'patch_length': 5,
        'stride': 2,
        'd_model': 128,
        'num_attention_heads': 8,
        'num_hidden_layers': 3
    }

    SENTIMENT_CONFIG = {
        'confidence_threshold': 0.6,
        'aggregation_window': 1,
        'corporate_events_weight': 0.3
    }

    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': os.path.join(DATA_PATHS['logs'], 'part_b.log'),
        'console_output': True
    }

    @classmethod
    def get_data_path(cls, data_type: str, filename: str = '') -> str:
        """
        Получить полный путь к файлу данных определенного типа.

        :param data_type: Тип данных ('raw', 'processed', 'results', 'logs')
        :param filename: Имя файла (опционально)
        :return: Полный путь к файлу или директории
        """
        base_path = cls.DATA_PATHS.get(data_type, cls.DATA_PATHS['raw'])
        if filename:
            return os.path.join(base_path, filename)
        return base_path

    @classmethod
    def create_directories(cls) -> None:
        """
        Создать все необходимые директории для работы системы.
        Создает папки для сырых данных, обработанных данных, результатов и логов.

        :return: None
        """
        for path in cls.DATA_PATHS.values():
            os.makedirs(path, exist_ok=True)

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Получить всю конфигурацию в виде словаря для сериализации или передачи.

        :return: Словарь со всеми настройками конфигурации
        """
        return {
            'data_paths': cls.DATA_PATHS,
            'data_files': cls.DATA_FILES,
            'finbert_config': cls.FINBERT_CONFIG,
            'patchtst_config': cls.PATCHTST_CONFIG,
            'sentiment_config': cls.SENTIMENT_CONFIG,
            'logging_config': cls.LOGGING_CONFIG
        }


def get_config():
    """
    Фабричная функция для получения объекта конфигурации.
    Обеспечивает обратную совместимость с предыдущими версиями.

    :return: Класс Config с настройками системы
    """
    return Config

__all__ = ['Config', 'get_config']