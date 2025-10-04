import os
import sys
import json
import warnings
import argparse
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from config import Config
from news_processing import process_news_for_patchtst
from patchtst_integration import prepare_data_for_patchtst
from patchtst_model import ForecastPatchTST
from stationarity_test import StationarityTester


class ForecastPipeline:
    """
    Полный пайплайн для хакатона FORECAST.

    Данный класс управляет полным пайплайном машинного обучения от загрузки данных
    до обучения модели и оценки, специально разработанным для прогнозирования
    финансовых временных рядов с использованием архитектуры PatchTST и
    интеграции анализа тональности.
    """

    def __init__(self):
        """
        Инициализирует пайплайн прогнозирования.

        Настраивает конфигурацию, хранилище результатов, заглушку модели и создает
        необходимые директории для выполнения пайплайна.
        """
        self.config = Config()
        self.results = {}
        self.model = None
        self.stationarity_tester = StationarityTester()

        Config.create_directories()

        print("FORECAST HACKATHON - FULL PIPELINE")
        print("=" * 60)
        print("Loading -> Aggregation -> ADF -> Training -> Test -> Statistics")
        print("=" * 60)

    def step1_load_data(self) -> bool:
        """
        Шаг 1: Загрузка сырых файлов данных.

        Загружает CSV файлы котировок и новостей, проверяет их структуру
        и сохраняет в словаре результатов для последующей обработки.

        Returns:
            bool: True если загрузка данных успешна, False иначе
        """
        print("STEP 1: DATA LOADING")
        print("-" * 40)

        try:
            candles_path = os.path.join(Config.DATA_PATHS['raw_participants'], Config.DATA_FILES['candles'])
            if not os.path.exists(candles_path):
                print(f"Candles file not found: {candles_path}")
                return False

            candles_df = pd.read_csv(candles_path)
            self.results['candles_raw'] = candles_df
            print(f"Candles loaded: {len(candles_df)} records, {candles_df['ticker'].nunique()} tickers")

            news_path = os.path.join(Config.DATA_PATHS['raw_participants'], Config.DATA_FILES['news'])
            if not os.path.exists(news_path):
                print(f"News file not found: {news_path}")
                return False

            news_df = pd.read_csv(news_path)
            self.results['news_raw'] = news_df
            print(f"News loaded: {len(news_df)} records")

            required_candle_cols = ['ticker', 'begin', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_candle_cols if col not in candles_df.columns]
            if missing_cols:
                print(f"Missing columns in candles: {missing_cols}")
                return False

            print(f"Data period: {candles_df['begin'].min()} - {candles_df['begin'].max()}")
            return True

        except Exception as e:
            print(f"Data loading error: {e}")
            return False

    def load_external_sentiment_features(self) -> bool:
        path = os.path.join(Config.DATA_PATHS['processed_participants'], Config.DATA_FILES['processed_sentiment'])

        if not os.path.exists(path):
                print(f"Sentiment features file not found: {path}")
                return False
        
        sentiment_features = pd.read_csv(path)
        self.results['sentiment_features'] = sentiment_features

        print(f"Sentiment features created: {len(sentiment_features)} records")
        print(f"Columns: {len(sentiment_features.columns)}")
        print(f"Tickers with sentiment: {sentiment_features['ticker'].nunique()}")

        if 'sentiment_mean' in sentiment_features.columns:
            sentiment_stats = sentiment_features['sentiment_mean'].describe()
            print(f"Sentiment statistics: mean={sentiment_stats['mean']:.3f}, std={sentiment_stats['std']:.3f}")

        if 'total_corporate_events' in sentiment_features.columns:
            corp_events_total = sentiment_features['total_corporate_events'].sum()
            print(f"Total corporate events: {int(corp_events_total)}")

        return True


    def step2_aggregate_news(self) -> bool:
        """
        Шаг 2: Агрегация новостей с анализом тональности.

        Обрабатывает новостные данные с использованием FinBERT и Enhanced Mock
        для анализа тональности, обнаруживает корпоративные события и создает
        временные признаки для входа модели.

        Returns:
            bool: True если агрегация новостей успешна, False иначе
        """
        print("\nSTEP 2: NEWS AGGREGATION AND SENTIMENT ANALYSIS")
        print("-" * 40)

        try:
            sentiment_features = process_news_for_patchtst()

            if sentiment_features.empty:
                print("Failed to process news")
                return False

            self.results['sentiment_features'] = sentiment_features

            print(f"Sentiment features created: {len(sentiment_features)} records")
            print(f"Columns: {len(sentiment_features.columns)}")
            print(f"Tickers with sentiment: {sentiment_features['ticker'].nunique()}")

            if 'sentiment_mean' in sentiment_features.columns:
                sentiment_stats = sentiment_features['sentiment_mean'].describe()
                print(f"Sentiment statistics: mean={sentiment_stats['mean']:.3f}, std={sentiment_stats['std']:.3f}")

            if 'total_corporate_events' in sentiment_features.columns:
                corp_events_total = sentiment_features['total_corporate_events'].sum()
                print(f"Total corporate events: {int(corp_events_total)}")

            return True

        except Exception as e:
            print(f"News processing error: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def step3_stationarity_test(self) -> bool:
        """
        Шаг 3: Тест стационарности ADF для временных рядов.

        Выполняет тесты Дики-Фуллера на уровнях цен и доходностях
        для каждого тикера для оценки стационарности и предоставления
        рекомендаций по предварительной обработке данных.

        Returns:
            bool: True если тест ADF выполнен успешно, False иначе
        """
        print("\nSTEP 3: ADF STATIONARITY TEST")
        print("-" * 40)

        try:
            candles_df = self.results['candles_raw']

            stationarity_results = self.stationarity_tester.test_multiple_series(
                candles_df, target_column='close'
            )

            self.results['stationarity_test'] = stationarity_results

            levels_stationary = stationarity_results['levels_stationary'].sum()
            returns_stationary = stationarity_results['returns_stationary'].sum()
            total_tickers = len(stationarity_results)

            print(f"ADF test completed for {total_tickers} tickers")
            print(f"Stationary price levels: {levels_stationary}/{total_tickers} ({levels_stationary/total_tickers*100:.1f}%)")
            print(f"Stationary returns: {returns_stationary}/{total_tickers} ({returns_stationary/total_tickers*100:.1f}%)")

            recommendations = stationarity_results['recommendation'].value_counts()
            print("Processing recommendations:")
            for rec, count in recommendations.items():
                print(f"   • {rec}: {count} tickers")

            return True

        except Exception as e:
            print(f"ADF test error: {e}")
            return False

    def step4_prepare_patchtst_data(self) -> bool:
        """
        Шаг 4: Подготовка данных для модели PatchTST.

        Интегрирует данные котировок и тональности, вычисляет технические индикаторы,
        создает последовательности специфичные для PatchTST с правильной структурой патчей
        и генерирует конфигурацию модели.

        Returns:
            bool: True если подготовка данных успешна, False иначе
        """
        print("\nSTEP 4: PATCHTST DATA PREPARATION")
        print("-" * 40)

        try:
            result = prepare_data_for_patchtst(
                candles_path=os.path.join(Config.DATA_PATHS['raw_participants'], Config.DATA_FILES['candles']),
                sentiment_path=os.path.join(Config.DATA_PATHS['processed_participants'], Config.DATA_FILES['processed_sentiment']),
                output_path=Config.DATA_PATHS['processed'],
                target_column='close'
            )

            if not result.get('success', False):
                print("Failed to prepare PatchTST data")
                return False

            self.results['patchtst_data'] = result

            sequences = result.get('sequences', {})
            if sequences and sequences.get('X') is not None:
                X = sequences['X']
                y = sequences['y']
                print(f"PatchTST sequences ready")
                print(f"Data shape: X={X.shape}, y={y.shape}")
                print(f"Number of features: {X.shape[-1]}")
                print(f"Unique tickers: {len(set(sequences.get('tickers', [])))}")

                print(f"X statistics: mean={np.mean(X):.4f}, std={np.std(X):.4f}")
                print(f"y statistics: mean={np.mean(y):.4f}, std={np.std(y):.4f}")
            else:
                print("Empty sequences for PatchTST")
                return False

            return True

        except Exception as e:
            print(f"Data preparation error: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def step5_train_model(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> bool:
        """
        Шаг 5: Обучение модели PatchTST.

        Разделяет данные на наборы обучения/валидации/тестирования, инициализирует
        модель PatchTST с правильной конфигурацией, обучает модель с ранней остановкой
        и сохраняет обученную модель.

        Args:
            train_ratio: Доля данных для обучения (по умолчанию: 0.7)
            val_ratio: Доля данных для валидации (по умолчанию: 0.15)

        Returns:
            bool: True если обучение успешно, False иначе
        """
        print("\nSTEP 5: PATCHTST MODEL TRAINING")
        print("-" * 40)

        try:
            patchtst_result = self.results['patchtst_data']
            sequences = patchtst_result.get('sequences', {})
            config = patchtst_result.get('config', {})

            if not sequences or sequences.get('X') is None:
                print("No data for training")
                return False

            X = sequences['X']
            y = sequences['y']

            n_samples = len(X)
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            print(f"Data split:")
            print(f"Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
            print(f"Val: {len(X_val)} samples ({val_ratio*100:.0f}%)")
            print(f"Test: {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")

            model_config = {
                'context_length': config.get('context_length', 20),
                'prediction_length': config.get('prediction_length', 1),
                'patch_length': config.get('patch_length', 5),
                'num_input_channels': config.get('num_input_channels', X.shape[-1]),
                'd_model': config.get('d_model', 128),
                'num_hidden_layers': config.get('num_hidden_layers', 3),
                'num_attention_heads': config.get('num_attention_heads', 8),
                'dropout': 0.1
            }

            print(f"Model configuration: {model_config}")

            self.model = ForecastPatchTST(model_config)

            print("Training started...")
            train_history = self.model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=50,
                batch_size=32,
                learning_rate=1e-4
            )

            self.results['train_history'] = train_history
            self.results['test_data'] = {'X_test': X_test, 'y_test': y_test}

            final_train_loss = train_history['train_loss'][-1]
            final_val_loss = train_history['val_loss'][-1]

            print(f"Training completed!")
            print(f"Final train loss: {final_train_loss:.4f}")
            print(f"Final val loss: {final_val_loss:.4f}")

            model_path = os.path.join(Config.DATA_PATHS['results'], 'patchtst_model.pth')
            self.model.save_model(model_path)
            print(f"Model saved: {model_path}")

            return True

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def step6_test_model(self) -> bool:
        """
        Шаг 6: Тестирование модели на тестовых данных.

        Оценивает обученную модель на отложенных тестовых данных, вычисляет
        метрики производительности включая MAE, RMSE и направленную точность,
        и сравнивает с базовыми моделями.

        Returns:
            bool: True если тестирование успешно, False иначе
        """
        print("\nSTEP 6: MODEL TESTING")
        print("-" * 40)

        try:
            if not self.model or not self.model.is_trained:
                print("Model not trained")
                return False

            test_data = self.results.get('test_data', {})
            X_test = test_data.get('X_test')
            y_test = test_data.get('y_test')

            if X_test is None or y_test is None:
                print("No test data available")
                return False

            print(f"Testing on {len(X_test)} samples...")

            predictions = self.model.predict(X_test)

            metrics = self.model.evaluate_model(X_test, y_test)

            self.results['test_predictions'] = predictions
            self.results['test_metrics'] = metrics

            print("Test results:")
            print(f"MAE (1-day): {metrics['mae_1d']:.4f}")
            print(f"RMSE (1-day): {metrics['rmse_1d']:.4f}")
            print(f"Direction Accuracy (1-day): {metrics['direction_accuracy_1d']:.3f}")

            if 'mae_20d' in metrics:
                print(f"MAE (20-day): {metrics['mae_20d']:.4f}")
                print(f"RMSE (20-day): {metrics['rmse_20d']:.4f}")
                print(f"Direction Accuracy (20-day): {metrics['direction_accuracy_20d']:.3f}")

            baseline_mae_1d = np.mean(np.abs(y_test[:, 0]))
            improvement_1d = (1 - metrics['mae_1d'] / baseline_mae_1d) * 100
            print(f"Improvement over baseline (1-day): {improvement_1d:.1f}%")

            return True

        except Exception as e:
            print(f"Testing error: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def step7_export_statistics(self) -> bool:
        """
        Шаг 7: Экспорт статистики и результатов.

        Создает комплексные отчеты включая статистику данных, производительность модели,
        результаты анализа стационарности и экспортирует все результаты в файлы JSON и CSV.
        Генерирует финальную сводку пайплайна.

        Returns:
            bool: True если экспорт успешен, False иначе
        """
        print("\nSTEP 7: STATISTICS EXPORT")
        print("-" * 40)

        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'full_forecast_pipeline_v1.0',
                'config': {
                    'data_paths': Config.DATA_PATHS,
                    'patchtst_config': self.results.get('patchtst_data', {}).get('config', {}),
                },
                'data_statistics': {},
                'model_performance': {},
                'stationarity_analysis': {},
                'files_created': [],
                'status': 'success'
            }

            if 'candles_raw' in self.results:
                candles_df = self.results['candles_raw']
                report['data_statistics']['candles'] = {
                    'records': len(candles_df),
                    'tickers': int(candles_df['ticker'].nunique()),
                    'date_range': {
                        'start': str(candles_df['begin'].min()),
                        'end': str(candles_df['begin'].max())
                    }
                }

            if 'sentiment_features' in self.results:
                sentiment_df = self.results['sentiment_features']
                report['data_statistics']['sentiment'] = {
                    'records': len(sentiment_df),
                    'features': len(sentiment_df.columns),
                    'tickers': int(sentiment_df['ticker'].nunique()) if 'ticker' in sentiment_df.columns else 0
                }

                if 'total_corporate_events' in sentiment_df.columns:
                    report['data_statistics']['corporate_events'] = int(sentiment_df['total_corporate_events'].sum())

            if 'test_metrics' in self.results:
                metrics = self.results['test_metrics']
                report['model_performance'] = {
                    'mae_1d': float(metrics['mae_1d']),
                    'rmse_1d': float(metrics['rmse_1d']),
                    'direction_accuracy_1d': float(metrics['direction_accuracy_1d'])
                }

                if 'mae_20d' in metrics:
                    report['model_performance'].update({
                        'mae_20d': float(metrics['mae_20d']),
                        'rmse_20d': float(metrics['rmse_20d']),
                        'direction_accuracy_20d': float(metrics['direction_accuracy_20d'])
                    })

            if 'stationarity_test' in self.results:
                stat_df = self.results['stationarity_test']
                report['stationarity_analysis'] = {
                    'total_tickers': len(stat_df),
                    'levels_stationary': int(stat_df['levels_stationary'].sum()),
                    'returns_stationary': int(stat_df['returns_stationary'].sum()),
                    'recommendations': stat_df['recommendation'].value_counts().to_dict()
                }

            output_files = [
                Config.get_data_path('processed', Config.DATA_FILES['processed_sentiment']),
                Config.get_data_path('processed', Config.DATA_FILES['patchtst_sequences']),
                Config.get_data_path('processed', Config.DATA_FILES['patchtst_config']),
                os.path.join(Config.DATA_PATHS['results'], 'patchtst_model.pth')
            ]
            report['files_created'] = [f for f in output_files if os.path.exists(f)]

            report_path = os.path.join(Config.DATA_PATHS['results'], 'forecast_pipeline_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"Report saved: {report_path}")

            if 'test_predictions' in self.results:
                predictions_df = pd.DataFrame(self.results['test_predictions'])
                predictions_path = os.path.join(Config.DATA_PATHS['results'], 'test_predictions.csv')
                predictions_df.to_csv(predictions_path, index=False)
                print(f"Predictions: {predictions_path}")

            if 'stationarity_test' in self.results:
                adf_path = os.path.join(Config.DATA_PATHS['results'], 'stationarity_test_results.csv')
                self.results['stationarity_test'].to_csv(adf_path, index=False)
                print(f"ADF results: {adf_path}")

            if 'train_history' in self.results:
                history_df = pd.DataFrame(self.results['train_history'])
                history_path = os.path.join(Config.DATA_PATHS['results'], 'training_history.csv')
                history_df.to_csv(history_path, index=False)
                print(f"Training history: {history_path}")

            print("\nFINAL STATISTICS:")
            print("-" * 40)
            print(f"Files created: {len(report['files_created'])}")
            if 'model_performance' in report and report['model_performance']:
                mae = report['model_performance']['mae_1d']
                acc = report['model_performance']['direction_accuracy_1d']
                print(f"Model MAE: {mae:.4f}")
                print(f"Direction accuracy: {acc:.3f}")

            self.results['final_report'] = report
            return True

        except Exception as e:
            print(f"Export error: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def run_full_pipeline(self) -> bool:
        """
        Выполнение полного пайплайна.

        Запускает все шаги пайплайна по порядку с правильной обработкой ошибок
        и отчетностью о прогрессе. Останавливает выполнение если любой шаг не удался.

        Returns:
            bool: True если все шаги выполнены успешно, False иначе
        """
        print("\nFULL PIPELINE EXECUTION")
        print("=" * 60)

        steps = [
            ("Data loading", self.step1_load_data),
            ("News aggregation", self.step2_aggregate_news),
            ("ADF test", self.step3_stationarity_test),
            ("PatchTST preparation", self.step4_prepare_patchtst_data),
            ("Model training", self.step5_train_model),
            ("Testing", self.step6_test_model),
            ("Create submission", self.create_submission_file),
            ("Statistics export", self.step7_export_statistics)
        ]

        for step_name, step_func in steps:
            print(f"\nExecuting: {step_name}")
            if not step_func():
                print(f"ERROR AT STEP: {step_name}")
                return False
            print(f"Completed: {step_name}")

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All steps executed successfully")
        print("Model trained and tested")
        print("Statistics exported")
        print("Ready for production!")
        print("=" * 60)

        return True
    
    def create_submission_file(self) -> bool:
        """Создание submission файла для хакатона FORECAST."""
        print("📄 CREATING SUBMISSION FILE")
        print("-" * 40)
        
        try:
            # Проверяем модель
            if not self.model or not self.model.is_trained:
                print("❌ Model not trained")
                return False
            
            # Получаем тестовые данные  
            test_data = self.results.get('test_data', {})
            X_test = test_data.get('X_test')
            
            if X_test is None:
                print("❌ No test data")
                return False
            
            # Генерируем предсказания
            predictions = self.model.predict(X_test)
            
            # Извлекаем метаданные
            sequences = self.results.get('patchtst_data', {}).get('sequences', {})
            test_tickers = sequences.get('tickers', [])[-len(X_test):]
            test_dates = sequences.get('dates', [])[-len(X_test):]
            
            # Создаем submission записи
            submission_data = []
            for i in range(len(predictions['return_1d'])):
                prob_1d = self._return_to_probability(predictions['return_1d'][i], '1d')
                prob_20d = self._return_to_probability(predictions['return_20d'][i], '20d')
                
                record = {
                    'ticker': test_tickers[i] if i < len(test_tickers) else f'T_{i}',
                    'day': test_dates[i] if i < len(test_dates) else f'2024-01-{15+i:02d}',
                    'return_1d': float(predictions['return_1d'][i]),
                    'prob_return_1d_positive': prob_1d,
                    'return_20d': float(predictions['return_20d'][i]), 
                    'prob_return_20d_positive': prob_20d
                }
                submission_data.append(record)
            
            # Сохраняем CSV
            submission_df = pd.DataFrame(submission_data)
            submission_path = Config.get_data_path('results', 'forecast_submission.csv')
            submission_df.to_csv(submission_path, index=False, float_format='%.6f')
            
            print(f"✅ Submission saved: {submission_path}")
            print(f"📊 Records: {len(submission_df)}, Tickers: {submission_df['ticker'].nunique()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _return_to_probability(self, return_value: float, horizon: str) -> float:
        """Конвертация доходности в вероятность роста."""
        import math
        scaling = {'1d': 10.0, '20d': 5.0}
        factor = scaling.get(horizon, 7.0)
        
        try:
            prob = 1 / (1 + math.exp(-return_value * factor))
        except OverflowError:
            prob = 0.95 if return_value > 0 else 0.05
        
        return max(0.05, min(0.95, prob))




def main():
    """
    Главная функция для выполнения пайплайна.
    
    Парсит аргументы командной строки, инициализирует пайплайн и выполняет
    запрошенные шаги с правильной обработкой ошибок и кодами выхода.
    """
    parser = argparse.ArgumentParser(description="FORECAST Hackathon - Full Pipeline")
    
    parser.add_argument(
        '--steps', 
        nargs='+',
        choices=['all', 'data', 'news', 'adf', 'patchtst', 'train', 'test', 'export', 'create_submission', 'load_external_sentiments'],
        default=['all'], 
        help='Which steps to execute (можно указать несколько через пробел)'
    )
    
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data proportion')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation data proportion')
    
    args = parser.parse_args()
    
    pipeline = ForecastPipeline()
    
    steps_map = {
        'data': pipeline.step1_load_data,
        'news': pipeline.step2_aggregate_news,
        'adf': pipeline.step3_stationarity_test,
        'patchtst': pipeline.step4_prepare_patchtst_data,
        'train': pipeline.step5_train_model,
        'test': pipeline.step6_test_model,
        'export': pipeline.step7_export_statistics,
        'create_submission': pipeline.create_submission_file,
        'load_external_sentiments': pipeline.load_external_sentiment_features
    }
    
    try:
        if 'all' in args.steps:
            success = pipeline.run_full_pipeline()
        else:
            success = True
            print(f"EXECUTING STEPS: {' -> '.join(args.steps)}")
            print("=" * 60)
            
            for i, step_name in enumerate(args.steps, 1):
                if step_name not in steps_map:
                    print(f"Unknown step: {step_name}")
                    success = False
                    break
                
                print(f"\nSTEP {i}/{len(args.steps)}: {step_name.upper()}")
                print("-" * 40)
                
                step_function = steps_map[step_name]
                if not step_function():
                    print(f"FAILED AT STEP: {step_name}")
                    success = False
                    break
                
                print(f"COMPLETED: {step_name}")
            
            if success:
                print("\nALL SELECTED STEPS COMPLETED SUCCESSFULLY!")
                print("=" * 60)
        
        if success:
            print("\nOPERATION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\nOPERATION COMPLETED WITH ERROR!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()