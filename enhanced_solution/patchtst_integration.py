import pandas as pd
import numpy as np
from typing import Dict
import os
from datetime import datetime, timedelta

from news_processing import SentimentAggregator


class PatchTSTDataIntegrator:
    """
    Интегратор данных для PatchTST модели прогнозирования временных рядов.
    Объединяет котировки акций, sentiment из новостей и технические индикаторы.
    """

    def __init__(self,
                 context_length: int = 20,
                 prediction_length: int = 1,
                 patch_length: int = 5,
                 stride: int = 2):
        """
        Инициализация интегратора с параметрами PatchTST архитектуры.

        :param context_length: Длина исторического контекста в днях
        :param prediction_length: Горизонт прогнозирования в днях
        :param patch_length: Размер патча для Transformer модели
        :param stride: Шаг между патчами
        :return: None
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride

        self.sentiment_aggregator = SentimentAggregator()

        print(f"PatchTST Data Integrator инициализирован:")
        print(f"Context: {context_length}, Prediction: {prediction_length}")
        print(f"Patch: {patch_length}, Stride: {stride}")

    def load_candles_data(self, data_path: str = "data/raw/",
                          filename: str = "train_candles.csv") -> pd.DataFrame:
        """
        Загрузка данных котировок акций с валидацией структуры и целевых переменных.

        :param data_path: Путь к папке с данными котировок
        :param filename: Имя файла с котировками
        :return: DataFrame с ценовыми данными и валидированной структурой
        """
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            print(f"Файл котировок не найден: {filepath}")
            return pd.DataFrame()

        try:
            candles_df = pd.read_csv(filepath)

            # Проверка наличия основных колонок
            expected_cols = ['begin', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in expected_cols if col not in candles_df.columns]

            if missing_cols:
                print(f"Отсутствуют колонки в котировках: {missing_cols}")

            if 'begin' in candles_df.columns:
                candles_df['begin'] = pd.to_datetime(candles_df['begin'])

            print(f"Загружено {len(candles_df)} записей котировок")
            if 'ticker' in candles_df.columns:
                print(f"Тикеров: {candles_df['ticker'].nunique()}")
            if 'begin' in candles_df.columns:
                print(f"Период: {candles_df['begin'].min()} - {candles_df['begin'].max()}")

            target_cols = ['target_return_1d', 'target_direction_1d', 'target_return_20d', 'target_direction_20d']
            found_targets = [col for col in target_cols if col in candles_df.columns]
            if found_targets:
                print(f"Target переменные найдены: {found_targets}")

            return candles_df

        except Exception as e:
            print(f"Ошибка загрузки котировок: {e}")
            return pd.DataFrame()

    def load_sentiment_features(self, data_path: str = "data/processed/",
                                filename: str = "processed_sentiment_features.csv") -> pd.DataFrame:
        """
        Загрузка обработанных sentiment признаков из новостных данных.

        :param data_path: Путь к папке с обработанными данными
        :param filename: Имя файла с sentiment признаками
        :return: DataFrame с sentiment признаками по дням и тикерам
        """
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            print(f"Файл sentiment не найден: {filepath}")
            return pd.DataFrame()

        try:
            sentiment_df = pd.read_csv(filepath)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

            print(f"Загружено {len(sentiment_df)} sentiment записей")
            print(f"Признаков: {len(sentiment_df.columns)}")
            if 'ticker' in sentiment_df.columns:
                print(f"Тикеров: {sentiment_df['ticker'].nunique()}")

            corporate_features = [col for col in sentiment_df.columns if 'corporate' in col.lower()]
            if corporate_features:
                print(f"Корпоративные признаки: {len(corporate_features)}")

            return sentiment_df

        except Exception as e:
            print(f"Ошибка загрузки sentiment: {e}")
            return pd.DataFrame()

    def merge_data_for_patchtst(self,
                                candles_df: pd.DataFrame,
                                sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Объединение ценовых данных и sentiment признаков для PatchTST с добавлением технических индикаторов.

        :param candles_df: DataFrame с котировками акций
        :param sentiment_df: DataFrame с sentiment признаками
        :return: Объединенный DataFrame со всеми признаками для обучения
        """
        if candles_df.empty:
            print("Пустые данные котировок")
            return pd.DataFrame()

        print("🔗 Объединение данных для PatchTST...")

        candles_prepared = candles_df.copy()
        if 'begin' in candles_prepared.columns:
            candles_prepared['date'] = candles_prepared['begin'].dt.date
        else:
            print("Колонка 'begin' не найдена в котировках")
            return pd.DataFrame()

        if sentiment_df.empty:
            print("Нет sentiment данных, используем только котировки с техническими индикаторами")
            merged_data = candles_prepared
        else:
            sentiment_prepared = sentiment_df.copy()
            sentiment_prepared['date'] = sentiment_prepared['date'].dt.date

            if 'ticker' in candles_prepared.columns and 'ticker' in sentiment_prepared.columns:
                merged_data = candles_prepared.merge(
                    sentiment_prepared,
                    on=['date', 'ticker'],
                    how='left'
                )
            else:
                print("Отсутствует колонка 'ticker' для объединения")
                merged_data = candles_prepared

            sentiment_columns = [col for col in merged_data.columns
                                 if any(keyword in col.lower() for keyword in
                                        ['sentiment', 'news', 'corporate', 'confidence'])]

            for col in sentiment_columns:
                if merged_data[col].dtype in ['int64', 'float64']:
                    merged_data[col] = merged_data[col].fillna(0.0)

        merged_data = self._add_technical_indicators(merged_data)

        print(f"Объединение завершено: {len(merged_data)} записей")

        numeric_features = len([c for c in merged_data.columns if merged_data[c].dtype in ['int64', 'float64']])
        if not sentiment_df.empty:
            sentiment_features = len([c for c in merged_data.columns if 'sentiment' in c.lower()])
            corporate_features = len([c for c in merged_data.columns if 'corporate' in c.lower()])
            print(f"Всего признаков: {numeric_features}")
            print(f"Sentiment признаков: {sentiment_features}")
            print(f"Корпоративных признаков: {corporate_features}")
        else:
            print(f"Всего признаков: {numeric_features} (только технические)")

        return merged_data

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов для каждого тикера отдельно.

        :param df: DataFrame с ценовыми данными
        :return: DataFrame с добавленными техническими индикаторами
        """
        df_enhanced = df.copy()

        print("Добавление технических индикаторов...")

        if 'close' not in df.columns:
            print("Колонка 'close' не найдена, пропускаем технические индикаторы")
            return df_enhanced

        if 'ticker' not in df.columns:
            print("Колонка 'ticker' не найдена, применяем индикаторы ко всем данным")
            df_enhanced = self._calculate_technical_indicators(df_enhanced)
        else:
            # Применение индикаторов для каждого тикера отдельно
            for ticker in df['ticker'].unique():
                mask = df_enhanced['ticker'] == ticker
                ticker_data = df_enhanced[mask].copy()

                if 'begin' in ticker_data.columns:
                    ticker_data = ticker_data.sort_values('begin')

                if len(ticker_data) == 0:
                    continue

                # Рассчёт технических индикаторов
                ticker_data_enhanced = self._calculate_technical_indicators(ticker_data)

                # Обновление основной DataFrame
                for col in ticker_data_enhanced.columns:
                    if col not in df_enhanced.columns:
                        df_enhanced[col] = 0.0
                    df_enhanced.loc[mask, col] = ticker_data_enhanced[col].values

        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        df_enhanced[numeric_cols] = df_enhanced[numeric_cols].fillna(0)

        print(f"Добавлены технические индикаторы")

        return df_enhanced

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет технических индикаторов для одного временного ряда.

        :param data: DataFrame с ценовыми данными одного актива
        :return: DataFrame с добавленными техническими индикаторами
        """
        df = data.copy()

        if 'close' not in df.columns:
            return df

        try:
            # Скользящие средние цены
            df['sma_5'] = df['close'].rolling(5, min_periods=1).mean()
            df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()

            # Доходности
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)

            # Волатильность
            df['volatility'] = df['return_1d'].rolling(10, min_periods=1).std()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            bb_window = 20
            df['bb_middle'] = df['close'].rolling(bb_window, min_periods=1).mean()
            bb_std = df['close'].rolling(bb_window, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        except Exception as e:
            print(f"Ошибка расчета технических индикаторов: {e}")

        return df

    def create_patchtst_sequences(self, merged_data: pd.DataFrame,
                                  target_column: str = 'close') -> Dict:
        """
        Создание временных последовательностей для обучения PatchTST модели.

        :param merged_data: Объединенные данные с признаками
        :param target_column: Название целевой колонки для прогнозирования
        :return: Словарь с массивами X, y, тикерами и датами
        """
        print(f"Создание последовательностей для PatchTST...")
        print(f"Target колонка: {target_column}")

        if merged_data.empty:
            return {'X': [], 'y': [], 'tickers': [], 'dates': []}

        exclude_columns = [
            'date', 'begin', 'ticker', target_column,
            'target_return_1d', 'target_direction_1d',
            'target_return_20d', 'target_direction_20d'
        ]

        feature_columns = [col for col in merged_data.columns
                           if merged_data[col].dtype in ['int64', 'float64'] and
                           col not in exclude_columns]

        print(f"Используется признаков: {len(feature_columns)}")

        # Категоризация признаков для анализа
        price_features = [col for col in feature_columns if col in ['open', 'high', 'low', 'close', 'volume']]
        technical_features = [col for col in feature_columns if
                              any(tech in col for tech in ['sma', 'rsi', 'macd', 'bb', 'volatility', 'return'])]
        sentiment_features = [col for col in feature_columns if 'sentiment' in col.lower()]
        corporate_features = [col for col in feature_columns if 'corporate' in col.lower()]

        print(f"Ценовые: {len(price_features)}, Технические: {len(technical_features)}")
        print(f"Sentiment: {len(sentiment_features)}, Корпоративные: {len(corporate_features)}")

        sequences = {'X': [], 'y': [], 'tickers': [], 'dates': []}

        if target_column not in merged_data.columns:
            print(f"Target колонка {target_column} не найдена в данных")
            available_targets = [col for col in merged_data.columns if 'target' in col.lower() or col in ['close']]
            if available_targets:
                print(f"Доступные target колонки: {available_targets}")
                target_column = available_targets[0]
                print(f"Используем: {target_column}")
            else:
                return sequences

        if 'ticker' in merged_data.columns:
            tickers_to_process = merged_data['ticker'].unique()
        else:
            tickers_to_process = ['ALL_DATA']
            merged_data = merged_data.copy()
            merged_data['ticker'] = 'ALL_DATA'

        for ticker in tickers_to_process:
            ticker_data = merged_data[merged_data['ticker'] == ticker].copy()

            if 'begin' in ticker_data.columns:
                ticker_data = ticker_data.sort_values('begin').reset_index(drop=True)
            else:
                ticker_data = ticker_data.reset_index(drop=True)

            min_length = self.context_length + self.prediction_length
            if len(ticker_data) < min_length:
                print(f"{ticker}: недостаточно данных ({len(ticker_data)} < {min_length})")
                continue

            valid_sequences = 0
            for i in range(len(ticker_data) - min_length + 1):
                try:
                    X_sequence = ticker_data[feature_columns].iloc[
                        i:i + self.context_length
                    ].values

                    y_sequence = ticker_data[target_column].iloc[
                        i + self.context_length:i + self.context_length + self.prediction_length
                    ].values

                    if (not np.isnan(X_sequence).any() and not np.isinf(X_sequence).any() and
                            not np.isnan(y_sequence).any() and not np.isinf(y_sequence).any() and
                            X_sequence.shape[0] == self.context_length and
                            len(y_sequence) == self.prediction_length):

                        sequences['X'].append(X_sequence)
                        sequences['y'].append(y_sequence)
                        sequences['tickers'].append(ticker)

                        # Дата
                        if 'begin' in ticker_data.columns:
                            date_val = ticker_data['begin'].iloc[i + self.context_length]
                        else:
                            date_val = i + self.context_length
                        sequences['dates'].append(date_val)

                        valid_sequences += 1

                except Exception as e:
                    continue

            if valid_sequences > 0:
                print(f"{ticker}: {valid_sequences} последовательностей")

        if sequences['X']:
            try:
                sequences['X'] = np.array(sequences['X'])
                sequences['y'] = np.array(sequences['y'])

                print(f"Создано последовательностей: {len(sequences['X'])}")
                print(f"Форма X: {sequences['X'].shape}")
                print(f"Форма y: {sequences['y'].shape}")

                # Статистика по признакам и target
                X_stats = {
                    'mean': np.mean(sequences['X']),
                    'std': np.std(sequences['X']),
                    'min': np.min(sequences['X']),
                    'max': np.max(sequences['X'])
                }

                y_stats = {
                    'mean': np.mean(sequences['y']),
                    'std': np.std(sequences['y']),
                    'min': np.min(sequences['y']),
                    'max': np.max(sequences['y'])
                }

                print(f"Статистика X: mean={X_stats['mean']:.4f}, std={X_stats['std']:.4f}")
                print(f"Статистика y: mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}")

            except Exception as e:
                print(f"Ошибка создания массивов: {e}")
                return {'X': [], 'y': [], 'tickers': [], 'dates': []}
        else:
            print("Не удалось создать последовательности")

        return sequences

    def get_patchtst_config(self, sequences: Dict, target_column: str = 'close') -> Dict:
        """
        Генерация конфигурации PatchTST модели на основе подготовленных данных.

        :param sequences: Словарь с последовательностями X, y
        :param target_column: Название целевой колонки
        :return: Словарь с конфигурацией модели
        """
        if not sequences or not sequences.get('X') or len(sequences['X']) == 0:
            print("Нет последовательностей для конфигурации")
            return {}

        X = sequences['X']
        y = sequences['y']
        n_features = X.shape[-1] if len(X.shape) > 2 else 0

        config = {
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'patch_length': min(self.patch_length, self.context_length),
            'stride': min(self.stride, self.patch_length),

            'num_input_channels': n_features,
            'd_model': 128,
            'num_attention_heads': 8,
            'num_hidden_layers': 3,
            'intermediate_size': 256,

            'dropout': 0.1,
            'attention_dropout': 0.1,
            'path_dropout': 0.1,

            'activation_function': 'gelu',

            'pooling_type': 'mean',
            'norm_type': 'batchnorm',
            'positional_encoding': 'sincos',

            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'early_stopping_patience': 10,

            'target_column': target_column,
            'target_statistics': {
                'mean': float(np.mean(y)) if len(y) > 0 else 0.0,
                'std': float(np.std(y)) if len(y) > 0 else 1.0,
                'min': float(np.min(y)) if len(y) > 0 else -1.0,
                'max': float(np.max(y)) if len(y) > 0 else 1.0
            }
        }

        print("Конфигурация PatchTST:")
        key_params = ['context_length', 'prediction_length', 'patch_length', 'num_input_channels', 'd_model',
                      'target_column']
        for key in key_params:
            if key in config:
                print(f"   {key}: {config[key]}")

        print(
            f"   Target статистика: mean={config['target_statistics']['mean']:.4f}, std={config['target_statistics']['std']:.4f}")

        return config

    def save_prepared_data(self, sequences: Dict, config: Dict,
                           output_path: str = "data/processed/") -> bool:
        """
        Сохранение подготовленных последовательностей и конфигурации для обучения модели.

        :param sequences: Словарь с последовательностями X, y, тикерами и датами
        :param config: Конфигурация модели
        :param output_path: Путь для сохранения файлов
        :return: True если сохранение успешно, False иначе
        """
        try:
            os.makedirs(output_path, exist_ok=True)

            # Сохраняем последовательности
            if sequences and sequences.get('X') is not None:
                sequences_file = os.path.join(output_path, 'patchtst_sequences.npz')
                np.savez(
                    sequences_file,
                    X=sequences['X'],
                    y=sequences['y'],
                    tickers=np.array(sequences['tickers']),
                    dates=np.array(sequences['dates'])
                )
                print(f"Последовательности сохранены: {sequences_file}")

            # Сохранение конфигурации
            if config:
                config_file = os.path.join(output_path, 'patchtst_config.json')
                import json
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
                print(f"Конфигурация сохранена: {config_file}")

            return True

        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return False


def prepare_data_for_patchtst(candles_path: str = "data/raw/train_candles.csv",
                              sentiment_path: str = "data/processed/processed_sentiment_features.csv",
                              output_path: str = "data/processed/",
                              target_column: str = None) -> Dict:
    """
    ГЛАВНАЯ ФУНКЦИЯ подготовки данных для обучения PatchTST модели.
    Координирует весь процесс от загрузки до сохранения готовых данных.

    :param candles_path: Путь к файлу с котировками акций
    :param sentiment_path: Путь к файлу с sentiment признаками
    :param output_path: Путь для сохранения подготовленных данных
    :param target_column: Целевая переменная (автоматически если None)
    :return: Словарь с результатами подготовки данных
    """
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ PATCHTST - ЧАСТЬ B")
    print("=" * 60)

    integrator = PatchTSTDataIntegrator(
        context_length=20,  # 20 дней истории
        prediction_length=1,  # Прогноз на 1 день
        patch_length=5,  # Патчи по 5 дней
        stride=2  # Шаг 2 дня
    )

    print("\n📊 Загрузка данных:")
    candles_df = integrator.load_candles_data(
        data_path=os.path.dirname(candles_path) if os.path.dirname(candles_path) else "data/raw/",
        filename=os.path.basename(candles_path)
    )

    sentiment_df = integrator.load_sentiment_features(
        data_path=os.path.dirname(sentiment_path) if os.path.dirname(sentiment_path) else "data/processed/",
        filename=os.path.basename(sentiment_path)
    )

    if candles_df.empty:
        print("Не удалось загрузить котировки")
        return {}

    if sentiment_df.empty:
        print("Sentiment данные не найдены, используем только котировки + технические индикаторы")
        sentiment_df = pd.DataFrame()

    if target_column is None:
        potential_targets = [col for col in candles_df.columns if 'target' in col.lower()]
        if potential_targets:
            target_column = potential_targets[0]
            print(f"Автоматически выбран target: {target_column}")
        else:
            target_column = 'close'  # По умолчанию
            print(f"Используем target по умолчанию: {target_column}")

    print("\nОбъединение данных:")
    merged_data = integrator.merge_data_for_patchtst(candles_df, sentiment_df)

    if merged_data.empty:
        print("Не удалось объединить данные")
        return {}

    print("\nСоздание последовательностей для PatchTST:")
    sequences = integrator.create_patchtst_sequences(merged_data, target_column)

    if not sequences or len(sequences.get('X', [])) == 0:
        print("Не удалось создать последовательности")
        return {}

    print("\nСоздание конфигурации:")
    config = integrator.get_patchtst_config(sequences, target_column)

    print("\nСохранение результатов:")
    success = integrator.save_prepared_data(sequences, config, output_path)

    if success:
        print("\n ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
        print(f"Готово для обучения PatchTST модели:")
        print(f"Последовательностей: {len(sequences['X'])}")
        print(f"Признаков: {sequences['X'].shape[-1]}")
        print(f"Тикеров: {len(set(sequences['tickers']))}")
        print(f"Target: {target_column}")

        print(f"\nSUPERVISED LEARNING ГОТОВ:")
        print(f"X.shape: {sequences['X'].shape}")
        print(f"y.shape: {sequences['y'].shape}")
    else:
        print("\nОшибка при сохранении данных")

    return {
        'sequences': sequences,
        'config': config,
        'merged_data': merged_data,
        'success': success,
        'target_column': target_column
    }


if __name__ == "__main__":
    result = prepare_data_for_patchtst()
    if result.get('success', False):
        print("Данные для PatchTST готовы!")
    else:
        print("Подготовка данных не удалась")