import os
import sys
import argparse
import json
import warnings
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from config import Config, get_config
from news_processing import process_news_for_patchtst
from patchtst_integration import prepare_data_for_patchtst


def setup_logging():
    """
    Настройка логирования для части B БЕЗ EMOJI для избежания Unicode ошибок.

    :return: Настроенный logger объект
    """
    import logging

    log_config = Config.LOGGING_CONFIG

    log_dir = os.path.dirname(log_config['file_path'])
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_config['file_path'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class PartBPipeline:
    """
    1. Валидация входных данных
    2. Обработка новостей с FinBERT
    3. Подготовка данных для PatchTST
    4. Валидация выходных данных
    5. Генерация итогового отчета
    """

    def __init__(self):
        """
        Инициализация пайплайна с настройкой логирования и создание директорий.
        Разделяет логи (без emoji) и консольные сообщения (с emoji).

        :return: None
        """
        self.logger = setup_logging()
        self.config = Config
        self.results = {}

        Config.create_directories()

        self.logger.info("Пайплайн части B инициализирован")
        self.logger.info(f"Конфигурация: {type(Config).__name__}")
        self.logger.info("Готов к обработке новостей и интеграции с PatchTST")

        print("Пайплайн части B инициализирован")
        print(f"Конфигурация: {type(Config).__name__}")
        print("Готов к обработке новостей и интеграции с PatchTST")

    def validate_input_data(self) -> bool:
        """
        Валидация входных данных: проверка существования файлов и их структуры
        Автоматически ищет альтернативные пути если основные не найдены.

        :return: True если все входные файлы найдены и валидны, False иначе
        """
        self.logger.info("Валидация входных данных...")
        print("Валидация входных данных...")

        # Проверяем существование файлов
        required_files = [
            (Config.get_data_path('raw', Config.DATA_FILES['candles']), "файл котировок"),
            (Config.get_data_path('raw', Config.DATA_FILES['news']), "файл новостей"),
        ]

        missing_files = []
        for filepath, description in required_files:
            if not os.path.exists(filepath):
                missing_files.append((filepath, description))
                self.logger.warning(f"Отсутствует {filepath} ({description})")
                print(f"Отсутствует {filepath} ({description})")

                # Проверяем альтернативные пути
                if 'train_news.csv' in filepath:
                    alternative_paths = [
                        'data/raw/participants/train_news.csv',
                        '../data/raw/participants/train_news.csv',
                        './data/raw/participants/train_news.csv'
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            self.logger.info(f"Найден альтернативный путь: {alt_path}")
                            print(f"Найден альтернативный путь: {alt_path}")
                            # Обновляем конфигурацию
                            Config.DATA_PATHS['raw'] = os.path.dirname(alt_path)
                            missing_files.remove((filepath, description))
                            break

                if 'train_candles.csv' in filepath:
                    alternative_paths = [
                        'data/raw/participants/train_candles.csv',
                        '../data/raw/participants/train_candles.csv',
                        './data/raw/participants/train_candles.csv'
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            self.logger.info(f"Найден альтернативный путь: {alt_path}")
                            print(f"Найден альтернативный путь: {alt_path}")
                            Config.DATA_PATHS['raw'] = os.path.dirname(alt_path)
                            missing_files.remove((filepath, description))
                            break

        if missing_files:
            self.logger.error(f"Отсутствуют необходимые файлы: {len(missing_files)}")
            self.logger.error("Проверьте структуру каталогов:")
            self.logger.error("   data/raw/participants/train_candles.csv")
            self.logger.error("   data/raw/participants/train_news.csv")

            print(f"Отсутствуют необходимые файлы: {len(missing_files)}")
            print("Проверьте структуру каталогов:")
            print("data/raw/participants/train_candles.csv")
            print("data/raw/participants/train_news.csv")
            return False

        try:
            candles_path = Config.get_data_path('raw', Config.DATA_FILES['candles'])
            candles_df = pd.read_csv(candles_path, nrows=5)

            basic_cols = ['ticker', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in basic_cols if col not in candles_df.columns]
            if missing_cols and len(missing_cols) > 2:
                self.logger.error(f"Критически не хватает колонок в котировках: {missing_cols}")
                print(f"Критически не хватает колонок в котировках: {missing_cols}")
                return False

            self.logger.info(
                f"Котировки: {len(candles_df)} строк (sample), {candles_df['ticker'].nunique() if 'ticker' in candles_df.columns else 'N/A'} тикеров")
            print(
                f"Котировки: {len(candles_df)} строк (sample), {candles_df['ticker'].nunique() if 'ticker' in candles_df.columns else 'N/A'} тикеров")

            news_path = Config.get_data_path('raw', Config.DATA_FILES['news'])
            news_df = pd.read_csv(news_path, nrows=5)

            if news_df.shape[1] < 3:
                self.logger.error(f"Недостаточно колонок в новостях: {news_df.shape[1]}")
                print(f"Недостаточно колонок в новостях: {news_df.shape[1]}")
                return False

            self.logger.info(f"Новости: {len(news_df)} строк (sample)")
            print(f"Новости: {len(news_df)} строк (sample)")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка валидации структуры данных: {e}")
            print(f"Ошибка валидации структуры данных: {e}")
            return False

    def run_news_processing(self) -> bool:
        """
        Запуск полной обработки новостей с FinBERT анализом и детекцией корпоративных событий.
        Включает sentiment анализ, извлечение тикеров, агрегацию и создание временных признаков.

        :return: True если обработка успешна, False при ошибках
        """
        self.logger.info("Запуск обработки новостей...")
        print("Запуск обработки новостей...")

        try:
            sentiment_features = process_news_for_patchtst(
                data_path=Config.DATA_PATHS['raw'],
                news_filename=Config.DATA_FILES['news']
            )

            if sentiment_features.empty:
                self.logger.error("Обработка новостей не удалась")
                print("Обработка новостей не удалась")
                return False

            self.results['sentiment_features'] = sentiment_features
            self.logger.info(f"Обработка завершена: {len(sentiment_features)} записей")
            print(f"Обработка завершена: {len(sentiment_features)} записей")

            if 'sentiment_mean' in sentiment_features.columns:
                sentiment_stats = sentiment_features['sentiment_mean'].describe()
                self.logger.info(
                    f"Статистика sentiment: mean={sentiment_stats['mean']:.3f}, std={sentiment_stats['std']:.3f}")
                print(f"Статистика sentiment: mean={sentiment_stats['mean']:.3f}, std={sentiment_stats['std']:.3f}")

            corporate_cols = [col for col in sentiment_features.columns if
                              'split' in col or 'dividend' in col or 'corporate' in col]
            if corporate_cols:
                self.logger.info(f"Корпоративных признаков: {len(corporate_cols)}")
                print(f"Корпоративных признаков: {len(corporate_cols)}")

            if 'total_corporate_events' in sentiment_features.columns:
                events_total = sentiment_features['total_corporate_events'].sum()
                self.logger.info(f"Всего корпоративных событий: {int(events_total)}")
                print(f"Всего корпоративных событий: {int(events_total)}")

            return True

        except Exception as e:
            self.logger.error(f"Ошибка обработки новостей: {e}")
            print(f"Ошибка обработки новостей: {e}")
            import traceback
            self.logger.error(f"Детали: {traceback.format_exc()}")
            return False

    def run_patchtst_preparation(self) -> bool:
        """
        Подготовка данных для PatchTST модели: объединение ценовых данных и sentiment,
        добавление технических индикаторов, создание временных последовательностей.

        :return: True если подготовка успешна, False при ошибках
        """
        self.logger.info("Подготовка данных для PatchTST...")
        print("Подготовка данных для PatchTST...")

        try:
            result = prepare_data_for_patchtst(
                candles_path=Config.get_data_path('raw', Config.DATA_FILES['candles']),
                sentiment_path=Config.get_data_path('processed', Config.DATA_FILES['processed_sentiment']),
                output_path=Config.DATA_PATHS['processed']
            )

            if not result.get('success', False):
                self.logger.error("Подготовка данных для PatchTST не удалась")
                print("Подготовка данных для PatchTST не удалась")
                return False

            self.results['patchtst_data'] = result

            sequences = result.get('sequences', {})
            if sequences and sequences.get('X') is not None:
                X = sequences['X']
                self.logger.info("Подготовка PatchTST завершена:")
                self.logger.info(f"   Последовательностей: {len(X)}")
                self.logger.info(f"   Форма входных данных: {X.shape}")
                self.logger.info(f"   Уникальных тикеров: {len(set(sequences.get('tickers', [])))}")

                print("Подготовка PatchTST завершена:")
                print(f"Последовательностей: {len(X)}")
                print(f"Форма входных данных: {X.shape}")
                print(f"Уникальных тикеров: {len(set(sequences.get('tickers', [])))}")

            config = result.get('config', {})
            if 'num_input_channels' in config:
                self.logger.info(f"   Входных признаков: {config['num_input_channels']}")
                print(f"Входных признаков: {config['num_input_channels']}")

            return True

        except Exception as e:
            self.logger.error(f"Ошибка подготовки PatchTST: {e}")
            print(f"Ошибка подготовки PatchTST: {e}")
            import traceback
            self.logger.error(f"Детали: {traceback.format_exc()}")
            return False

    def validate_output_data(self) -> bool:
        """
        Валидация выходных данных: проверка размерностей массивов, отсутствия NaN/Infinity,
        соответствия размеров входных и выходных данных для PatchTST.

        :return: True если все выходные данные корректны, False иначе
        """
        self.logger.info("Валидация выходных данных...")
        print("Валидация выходных данных...")

        try:
            sentiment_path = Config.get_data_path('processed', Config.DATA_FILES['processed_sentiment'])
            if not os.path.exists(sentiment_path):
                self.logger.error(f"Отсутствует файл sentiment: {sentiment_path}")
                print(f"Отсутствует файл sentiment: {sentiment_path}")
                return False

            sentiment_df = pd.read_csv(sentiment_path)
            if sentiment_df.empty:
                self.logger.error("Пустой файл sentiment")
                print("Пустой файл sentiment")
                return False

            self.logger.info(f"Sentiment файл: {len(sentiment_df)} записей, {len(sentiment_df.columns)} колонок")
            print(f"Sentiment файл: {len(sentiment_df)} записей, {len(sentiment_df.columns)} колонок")

            sequences_path = Config.get_data_path('processed', Config.DATA_FILES['patchtst_sequences'])
            config_path = Config.get_data_path('processed', Config.DATA_FILES['patchtst_config'])

            if not os.path.exists(sequences_path):
                self.logger.error(f"Отсутствует файл последовательностей: {sequences_path}")
                print(f"Отсутствует файл последовательностей: {sequences_path}")
                return False

            if not os.path.exists(config_path):
                self.logger.error(f"Отсутствует файл конфигурации: {config_path}")
                print(f"Отсутствует файл конфигурации: {config_path}")
                return False

            sequences_data = np.load(sequences_path)
            X = sequences_data['X']
            y = sequences_data['y']

            if len(X.shape) != 3:
                self.logger.error(f"Неправильная размерность X: {X.shape}, ожидается 3D")
                print(f"Неправильная размерность X: {X.shape}, ожидается 3D")
                return False

            if len(y.shape) != 2:
                self.logger.error(f"Неправильная размерность y: {y.shape}, ожидается 2D")
                print(f"Неправильная размерность y: {y.shape}, ожидается 2D")
                return False

            if X.shape[0] != y.shape[0]:
                self.logger.error(f"Несоответствие размеров: X={X.shape[0]}, y={y.shape[0]}")
                print(f"Несоответствие размеров: X={X.shape[0]}, y={y.shape[0]}")
                return False

            if np.isnan(X).any() or np.isinf(X).any():
                self.logger.error("X содержит NaN или Inf значения")
                print("X содержит NaN или Inf значения")
                return False

            if np.isnan(y).any() or np.isinf(y).any():
                self.logger.error("y содержит NaN или Inf значения")
                print("y содержит NaN или Inf значения")
                return False

            self.logger.info(f"Данные PatchTST: X={X.shape}, y={y.shape}")
            self.logger.info("Все данные валидны")
            print(f"Данные PatchTST: X={X.shape}, y={y.shape}")
            print("Все данные валидны")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка валидации выходных данных: {e}")
            print(f"Ошибка валидации выходных данных: {e}")
            return False

    def generate_report(self) -> Dict:
        """
        Генерация детального JSON отчета о работе пайплайна с статистиками,
        списком созданных файлов и метрик обработки.

        :return: Словарь с полным отчетом о выполнении пайплайна
        """
        self.logger.info("Генерация итогового отчета...")
        print("Генерация итогового отчета...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'version': 'unicode_fixed_enhanced_with_corporate_events',
            'config': type(Config).__name__,
            'status': 'success',
            'steps_completed': [],
            'data_stats': {},
            'files_created': [],
            'errors': [],
            'corporate_events': {}
        }

        try:
            # Статистика sentiment обработки
            if 'sentiment_features' in self.results:
                sentiment_df = self.results['sentiment_features']
                report['data_stats']['sentiment'] = {
                    'records': len(sentiment_df),
                    'features': len(sentiment_df.columns),
                    'tickers': sentiment_df['ticker'].nunique() if 'ticker' in sentiment_df.columns else 0,
                    'date_range': {
                        'start': str(sentiment_df['date'].min()) if 'date' in sentiment_df.columns else None,
                        'end': str(sentiment_df['date'].max()) if 'date' in sentiment_df.columns else None
                    }
                }

                # Статистика корпоративных событий
                corporate_cols = [col for col in sentiment_df.columns if
                                  'split' in col or 'dividend' in col or 'corporate' in col]
                report['corporate_events']['features_count'] = len(corporate_cols)

                if 'total_corporate_events' in sentiment_df.columns:
                    report['corporate_events']['total_events'] = int(sentiment_df['total_corporate_events'].sum())

            # Статистика PatchTST подготовки
            if 'patchtst_data' in self.results:
                patchtst_result = self.results['patchtst_data']
                sequences = patchtst_result.get('sequences', {})
                if sequences and sequences.get('X') is not None:
                    X = sequences['X']
                    report['data_stats']['patchtst'] = {
                        'sequences': len(X),
                        'features': X.shape[-1],
                        'context_length': X.shape[1],
                        'tickers': len(set(sequences.get('tickers', [])))
                    }

            output_files = [
                Config.get_data_path('processed', Config.DATA_FILES['processed_sentiment']),
                Config.get_data_path('processed', Config.DATA_FILES['patchtst_sequences']),
                Config.get_data_path('processed', Config.DATA_FILES['patchtst_config']),
                Config.get_data_path('processed', 'processed_news_with_corporate_events.csv')
            ]

            report['files_created'] = [f for f in output_files if os.path.exists(f)]

            report_path = Config.get_data_path('results', 'part_b_unicode_fixed_report.json')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Отчет сохранен: {report_path}")
            print(f"Отчет сохранен: {report_path}")

            return report

        except Exception as e:
            self.logger.error(f"Ошибка генерации отчета: {e}")
            print(f"Ошибка генерации отчета: {e}")
            report['status'] = 'error'
            report['errors'].append(str(e))
            return report

    def run_full_pipeline(self, validate_inputs: bool = True, validate_outputs: bool = True) -> bool:
        """
        Запуск полного пайплайна части B через все 5 этапов с опциональной валидацией.
        Контролирует выполнение всех этапов и генерирует итоговый отчет.

        :param validate_inputs: Проводить ли валидацию входных данных
        :param validate_outputs: Проводить ли валидацию выходных данных
        :return: True если весь пайплайн выполнен успешно, False при ошибках
        """
        self.logger.info("ЗАПУСКПАЙПЛАЙНА ЧАСТИ B")
        self.logger.info("=" * 60)
        print("ЗАПУСКПАЙПЛАЙНА ЧАСТИ B")
        print("=" * 60)

        success = True

        try:
            if validate_inputs:
                self.logger.info("1. Валидация входных данных")
                print("1️⃣ Валидация входных данных")
                if not self.validate_input_data():
                    return False

            self.logger.info("2. Обработка новостей с FinBERT")
            print("2️⃣ Обработка новостей с FinBERT")
            if not self.run_news_processing():
                success = False

            self.logger.info("3. Подготовка данных для PatchTST")
            print("3️⃣ Подготовка данных для PatchTST")
            if not self.run_patchtst_preparation():
                success = False

            if validate_outputs and success:
                self.logger.info("4. Валидация выходных данных")
                print("4️⃣ Валидация выходных данных")
                if not self.validate_output_data():
                    success = False

            self.logger.info("5. Генерация отчета")
            print("5️⃣ Генерация отчета")
            report = self.generate_report()

            if success:
                self.logger.info("=" * 60)
                self.logger.info("ПАЙПЛАЙН ЧАСТИ B ЗАВЕРШЕН УСПЕШНО!")
                self.logger.info("Обработка новостей завершена")
                self.logger.info("Интеграция с частями A и C готова")
                self.logger.info("Данные готовы для обучения PatchTST")
                self.logger.info("=" * 60)

                print("=" * 60)
                print("ПАЙПЛАЙН ЧАСТИ B ЗАВЕРШЕН УСПЕШНО!")
                print("Обработка новостей завершена")
                print("Интеграция с частями A и C готова")
                print("Данные готовы для обучения PatchTST")
                print("=" * 60)
            else:
                self.logger.error("=" * 60)
                self.logger.error("ПАЙПЛАЙН ЗАВЕРШЕН С ОШИБКАМИ")
                self.logger.error("=" * 60)

                print("=" * 60)
                print("ПАЙПЛАЙН ЗАВЕРШЕН С ОШИБКАМИ")
                print("=" * 60)

            return success

        except Exception as e:
            self.logger.error(f"Критическая ошибка пайплайна: {e}")
            print(f"Критическая ошибка пайплайна: {e}")
            import traceback
            self.logger.error(f"Детали: {traceback.format_exc()}")
            return False


def main():
    """
    Главная функция запуска пайплайна части B с обработкой аргументов командной строки.
    Поддерживает различные режимы работы и настройки конфигурации.

    :return: None (завершает программу с exit code 0 при успехе, 1 при ошибке)
    """
    parser = argparse.ArgumentParser(
        description="пайплайн части B хакатона FORECAST - обработка новостей + PatchTST"
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'news-only', 'patchtst-only'],
        default='full',
        help='Режим запуска пайплайна'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Пропустить валидацию данных'
    )

    parser.add_argument(
        '--config-mode',
        choices=['development', 'production'],
        default='development',
        help='Режим конфигурации'
    )

    args = parser.parse_args()

    if args.config_mode:
        os.environ['FORECAST_MODE'] = args.config_mode

    global Config
    Config = get_config()

    pipeline = PartBPipeline()

    try:
        if args.mode == 'full':
            success = pipeline.run_full_pipeline(
                validate_inputs=not args.skip_validation,
                validate_outputs=not args.skip_validation
            )
        elif args.mode == 'news-only':
            success = pipeline.run_news_processing()
        elif args.mode == 'patchtst-only':
            success = pipeline.run_patchtst_preparation()
        else:
            print(f"Неизвестный режим: {args.mode}")
            success = False

        if success:
            print("\nРЕЗУЛЬТАТ: УСПЕХ")
            print("пайплайн части B завершен успешно")
            print("Готов к интеграции с частями A и C")
            sys.exit(0)
        else:
            print("\nРЕЗУЛЬТАТ: ОШИБКА")
            print("Пайплайн части B завершен с ошибками")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nВыполнение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        print(f"Детали: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()