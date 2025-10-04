import os
from typing import Dict, Any

import torch


class Config:
    """
    Центральный класс конфигурации для системы обработки новостей части B.
    Содержит пути к файлам, настройки моделей и параметры обработки.
    """

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATHS = {
        'raw': os.path.join(BASE_PATH, '..', 'data', 'raw'),
        'raw_participants': os.path.join(BASE_PATH, '..', 'data', 'raw', 'participants'),
        'processed_participants': os.path.join(BASE_PATH, '..', 'data', 'processed', 'participants'),
        'processed': os.path.join(BASE_PATH, '..', 'data', 'processed'),
        'results': os.path.join(BASE_PATH, '..', 'data', 'results'),
        'logs': os.path.join(BASE_PATH, '..', 'logs')
    }

    DATA_FILES = {
        'candles': 'train_candles.csv',
        'news': 'train_news.csv',
        'processed_sentiment': 'processed_sentiment_features.csv',
        'processed_news_with_corporate_events': 'processed_news_with_corporate_events.csv',
        'patchtst_sequences': 'patchtst_sequences.npz',
        'patchtst_config': 'patchtst_config.json'
    }

    ALTERNATIVE_PATHS = [
        'data/raw/participants/',
        '../data/raw/',
        'data/',
        './'
    ]

    FINBERT_CONFIG = {
        'model_name': 'ProsusAI/finbert',
        'max_length': 512,
        'batch_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'truncation': True,
        'padding': True,
        'return_tensors': 'pt'
    }

    # Настройки PatchTST
    PATCHTST_CONFIG = {
        'context_length': 96,
        'prediction_length': 20,
        'patch_length': 12,
        'stride': 6,
        'd_model': 256,
        'num_attention_heads': 8,
        'num_hidden_layers': 4,
        'num_input_channels': 15,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'path_dropout': 0.1,
        'ff_dropout': 0.1,
        'scaling': 'std',
        'loss': 'mse'
    }

    VALIDATION_CONFIG = {
        'min_news_columns': 3,
        'min_candles_columns': 6,
        'sample_rows_for_validation': 5,
        'min_sequence_length': 30,
        'max_missing_ratio': 0.1,
        'outlier_std_threshold': 3.0
    }

    TRAINING_CONFIG = {
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 5,
        'gradient_clip_max_norm': 1.0,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau'
    }

    RANDOM_SEEDS = {
        'torch': 52,
        'numpy': 52,
        'python': 52,
        'tf': 52
    }

    SENTIMENT_CONFIG = {
        'confidence_threshold': 0.6,
        'aggregation_window': 1,
        'corporate_events_weight': 0.3,
        'split_adjustment': 0.2,
        'dividend_adjustment': 0.3,
        'merger_adjustment': 0.4,
        'spinoff_adjustment': 0.25,
        'rolling_windows': [3, 7, 14],
        'text_max_length': 450,
        'text_truncate_start': 200,
        'text_truncate_end': 200
    }

    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': os.path.join(DATA_PATHS['logs'], 'part_b.log'),
        'console_output': True
    }

    RANDOM_SEEDS = {
        'torch': 52,
        'numpy': 52,
        'python': 52
    }

    TECHNICAL_INDICATORS_CONFIG = {
        'sma_windows': [5, 20],
        'ema_spans': [12, 26],
        'rsi_period': 14,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'volatility_window': 10,
        'atr_period': 14,
        'stochastic_k_period': 14,
        'stochastic_d_period': 3
    }

    RUSSIAN_TICKERS = {
        'known_tickers': [
            'SBER', 'GAZP', 'YNDX', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'PLZL',
            'TATN', 'MGNT', 'CHMF', 'NLMK', 'AFLT', 'VTBR', 'POLY', 'RTKM',
            'FIVE', 'FIXP', 'IRAO', 'DSKY', 'QIWI', 'MAIL', 'OZON', 'ALRS',
            'MTSS', 'RUAL', 'SNGS', 'TRNFP', 'UPRO', 'FEES', 'MOEX', 'TCSG'
        ],
        'company_to_ticker': {
            'сбербанк': 'SBER', 'сбер': 'SBER', 'sberbank': 'SBER', 'sber': 'SBER',
            'газпром': 'GAZP', 'gazprom': 'GAZP',
            'яндекс': 'YNDX', 'yandex': 'YNDX',
            'лукойл': 'LKOH', 'lukoil': 'LKOH',
            'норникель': 'GMKN', 'новатэк': 'NVTK',
            'роснефть': 'ROSN', 'rosneft': 'ROSN',
            'полюс': 'PLZL', 'polyus': 'PLZL',
            'татнефть': 'TATN', 'tatneft': 'TATN',
            'магнит': 'MGNT', 'magnit': 'MGNT',
            'северсталь': 'CHMF', 'severstal': 'CHMF',
            'нлмк': 'NLMK', 'nlmk': 'NLMK',
            'аэрофлот': 'AFLT', 'aeroflot': 'AFLT',
            'втб': 'VTBR', 'vtb': 'VTBR',
            'полиметалл': 'POLY', 'polymetal': 'POLY',
            'ростелеком': 'RTKM', 'rostelecom': 'RTKM'
        }
    }

    PREDICTION_CONFIG = {
        'horizons': [1, 20],  # 1-day, 20-day
        'probability_scaling': {
            '1d': 10,  # коэффициент для sigmoid
            '20d': 5
        },
        'probability_bounds': [0.05, 0.95],  # минимум и максимум для clip
        'return_calculation_method': 'log',   # 'simple' или 'log'
        'target_columns': ['return_1d', 'return_20d']
    }

    STATIONARITY_CONFIG = {      # Для stationarity_test.py
        'significance_level': 0.05,
        'min_series_length': 10,
        'autolag': 'AIC'
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
    
    @classmethod
    def set_random_seeds(cls) -> None:          # ✅ ДОБАВИТЬ МЕТОД
        """Установить все random seeds для воспроизводимости."""
        import random
        import numpy as np
        
        random.seed(cls.RANDOM_SEEDS['python'])
        np.random.seed(cls.RANDOM_SEEDS['numpy'])
        
        try:
            import torch
            torch.manual_seed(cls.RANDOM_SEEDS['torch'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cls.RANDOM_SEEDS['torch'])
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            tf.random.set_seed(cls.RANDOM_SEEDS['tf'])
        except ImportError:
            pass


def get_config():
    """
    Фабричная функция для получения объекта конфигурации.
    Обеспечивает обратную совместимость с предыдущими версиями.

    :return: Класс Config с настройками системы
    """
    return Config

__all__ = ['Config', 'get_config']