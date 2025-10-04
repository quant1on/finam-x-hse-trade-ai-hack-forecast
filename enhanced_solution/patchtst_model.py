import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction
from transformers import PatchTSTModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class ForecastPatchTST:
    """PatchTST модель для задачи FORECAST"""
    
    def __init__(self, 
                 ModelConfig: Dict):
        # Конфигурация PatchTST
        self.config = PatchTSTConfig(**ModelConfig)
        
        self.model = None
        self.is_trained = False
        
        # Для воспроизводимости
        torch.manual_seed(52)
        np.random.seed(52)
    
    def _create_model(self):
        """Создание модели PatchTST"""
        if self.model is None:
            # Используем PatchTSTForPrediction для forecasting задач
            self.model = PatchTSTForPrediction(self.config)
    
    def prepare_data_for_patchtst(self, X: np.ndarray, y: np.ndarray = None) -> Dict:
        """
        Подготовка данных в формате PatchTST
        X: (batch_size, sequence_length, num_features)
        y: (batch_size, 2) - [return_1d, return_20d]
        """
        # PatchTST ожидает данные в формате (batch_size, sequence_length, num_channels)
        # Наши данные уже в правильном формате
        
        # Конвертируем в torch tensors
        past_values = torch.tensor(X, dtype=torch.float32)
        
        data_dict = {
            'past_values': past_values
        }
        
        if y is not None:
            # Для обучения нам нужны future values
            # PatchTST прогнозирует на prediction_length шагов вперед
            # Но нам нужны только специфические горизонты (1d, 20d)
            
            # Создаем "фиктивные" future_values для compatibility
            batch_size = X.shape
            # Повторяем последний таргет для всех prediction_length шагов
            future_values = torch.zeros(batch_size, self.prediction_length, self.num_input_channels)
            
            data_dict['future_values'] = future_values
            data_dict['targets'] = torch.tensor(y, dtype=torch.float32)
        
        return data_dict
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 1e-4) -> Dict:
        """Обучение PatchTST модели"""
        
        print(f"🚀 Начинаем обучение PatchTST модели")
        print(f"   Размер обучающей выборки: {X_train.shape}")
        print(f"   Размер валидационной выборки: {X_val.shape}")
        print(f"   Context length: {self.context_length}")
        print(f"   Prediction length: {self.prediction_length}")
        print(f"   Patch length: {self.patch_length}")
        
        self._create_model()
        
        train_data = self.prepare_data_for_patchtst(X_train, y_train)
        val_data = self.prepare_data_for_patchtst(X_val, y_val)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Для tracking истории обучения
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae_1d': [],
            'val_mae_1d': [],
            'train_mae_20d': [],
            'val_mae_20d': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_mae_1d = 0.0
            train_mae_20d = 0.0
            
            # Простая batch обработка (можно оптимизировать с DataLoader)
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                batch_data = self.prepare_data_for_patchtst(batch_X, batch_y)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch_data.items() if k != 'targets'})
                
                # Кастомная loss function для наших специфических горизонтов
                loss, mae_1d, mae_20d = self._calculate_custom_loss(outputs, batch_data['targets'])
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_mae_1d += mae_1d
                train_mae_20d += mae_20d
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_mae_1d = 0.0
            val_mae_20d = 0.0
            
            with torch.no_grad():
                val_n_batches = (len(X_val) + batch_size - 1) // batch_size
                
                for batch_idx in range(val_n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(X_val))
                    
                    batch_X = X_val[start_idx:end_idx]
                    batch_y = y_val[start_idx:end_idx]
                    
                    batch_data = self.prepare_data_for_patchtst(batch_X, batch_y)
                    
                    outputs = self.model(**{k: v for k, v in batch_data.items() if k != 'targets'})
                    loss, mae_1d, mae_20d = self._calculate_custom_loss(outputs, batch_data['targets'])
                    
                    val_loss += loss.item()
                    val_mae_1d += mae_1d
                    val_mae_20d += mae_20d
            
            # Усредняем метрики
            train_loss /= n_batches
            train_mae_1d /= n_batches
            train_mae_20d /= n_batches
            val_loss /= val_n_batches
            val_mae_1d /= val_n_batches
            val_mae_20d /= val_n_batches
            
            # Сохраняем историю
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae_1d'].append(train_mae_1d)
            history['val_mae_1d'].append(val_mae_1d)
            history['train_mae_20d'].append(train_mae_20d)
            history['val_mae_20d'].append(val_mae_20d)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(self.model.state_dict(), 'best_patchtst_model.pth')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val MAE 1d: {val_mae_1d:.4f}, Val MAE 20d: {val_mae_20d:.4f}")
            
            # Early stopping
            if patience_counter >= 10:
                print(f"Early stopping на epoch {epoch}")
                break
        
        # Загружаем лучшую модель
        self.model.load_state_dict(torch.load('best_patchtst_model.pth'))
        self.is_trained = True
        
        print("✅ Обучение завершено!")
        return history
    
    def _calculate_custom_loss(self, outputs, targets):
        """
        Кастомная loss функция для наших специфических горизонтов прогнозирования
        outputs: результат PatchTST модели
        targets: (batch_size, 2) - [return_1d, return_20d]
        """
        # PatchTST возвращает прогнозы на prediction_length шагов
        # Нам нужны только первый шаг (1d) и последний шаг (20d)
        predictions = outputs.prediction_outputs  # (batch_size, prediction_length, num_channels)
        
        # Извлекаем предсказания для целевых признаков (например, close price)
        # Берем первый канал как основной (или можно сделать average по каналам)
        pred_1d = predictions[:, 0, 0]   # Первый временной шаг, первый канал
        pred_20d = predictions[:, -1, 0]  # Последний временной шаг, первый канал
        
        # Целевые значения
        target_1d = targets[:, 0]
        target_20d = targets[:, 1]
        
        # MSE loss для каждого горизонта
        loss_1d = nn.functional.mse_loss(pred_1d, target_1d)
        loss_20d = nn.functional.mse_loss(pred_20d, target_20d)
        
        # Комбинированная loss (можно настроить веса)
        total_loss = loss_1d + loss_20d
        
        # MAE для мониторинга
        mae_1d = nn.functional.l1_loss(pred_1d, target_1d).item()
        mae_20d = nn.functional.l1_loss(pred_20d, target_20d).item()
        
        return total_loss, mae_1d, mae_20d
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Предсказания PatchTST модели"""
        if not self.is_trained:
            raise ValueError("Модель не обучена! Вызовите train() сначала.")
        
        self.model.eval()
        predictions = {'return_1d': [], 'return_20d': [], 'prob_1d': [], 'prob_20d': []}
        
        with torch.no_grad():
            # Батчевая обработка
            batch_size = 32
            n_batches = (len(X) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                
                batch_X = X[start_idx:end_idx]
                batch_data = self.prepare_data_for_patchtst(batch_X)
                
                # Получаем предсказания
                outputs = self.model(**batch_data)
                forecast = outputs.prediction_outputs  # (batch_size, prediction_length, num_channels)
                
                # Извлекаем нужные горизонты
                pred_1d = forecast[:, 0, 0].cpu().numpy()    # 1-day horizon
                pred_20d = forecast[:, -1, 0].cpu().numpy()  # 20-day horizon
                
                # Вероятности роста (sigmoid от предсказаний)
                prob_1d = torch.sigmoid(torch.tensor(pred_1d * 10)).cpu().numpy()  # Scaling factor
                prob_20d = torch.sigmoid(torch.tensor(pred_20d * 5)).cpu().numpy()
                
                # Clip вероятности в разумные пределы
                prob_1d = np.clip(prob_1d, 0.1, 0.9)
                prob_20d = np.clip(prob_20d, 0.1, 0.9)
                
                predictions['return_1d'].extend(pred_1d)
                predictions['return_20d'].extend(pred_20d)
                predictions['prob_1d'].extend(prob_1d)
                predictions['prob_20d'].extend(prob_20d)
        
        # Конвертируем в numpy arrays
        for key in predictions:
            predictions[key] = np.array(predictions[key])
        
        return predictions
    
    def evaluate_model(self, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Оценка качества PatchTST модели"""
        
        # Получаем предсказания
        predictions = self.predict(X_test)
        
        # Вычисляем метрики
        metrics = {}
        
        # MAE для доходностей
        metrics['mae_1d'] = mean_absolute_error(y_test[:, 0], predictions['return_1d'])
        metrics['mae_20d'] = mean_absolute_error(y_test[:, 1], predictions['return_20d'])
        
        # RMSE для доходностей
        metrics['rmse_1d'] = np.sqrt(mean_squared_error(y_test[:, 0], predictions['return_1d']))
        metrics['rmse_20d'] = np.sqrt(mean_squared_error(y_test[:, 1], predictions['return_20d']))
        
        # Direction Accuracy
        true_direction_1d = (y_test[:, 0] > 0).astype(int)
        pred_direction_1d = (predictions['return_1d'] > 0).astype(int)
        metrics['direction_accuracy_1d'] = np.mean(true_direction_1d == pred_direction_1d)
        
        true_direction_20d = (y_test[:, 1] > 0).astype(int)
        pred_direction_20d = (predictions['return_20d'] > 0).astype(int)
        metrics['direction_accuracy_20d'] = np.mean(true_direction_20d == pred_direction_20d)
        
        # Brier Score для вероятностей
        true_prob_1d = (y_test[:, 0] > 0).astype(float)
        true_prob_20d = (y_test[:, 1] > 0).astype(float)
        
        metrics['brier_1d'] = np.mean((true_prob_1d - predictions['prob_1d']) ** 2)
        metrics['brier_20d'] = np.mean((true_prob_20d - predictions['prob_20d']) ** 2)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Сохранение модели"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'num_input_channels': self.num_input_channels
            }, filepath)
            print(f"💾 PatchTST модель сохранена: {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка модели"""
        checkpoint = torch.load(filepath)
        
        # Восстанавливаем конфигурацию
        self.config = checkpoint['config']
        self.context_length = checkpoint['context_length']
        self.prediction_length = checkpoint['prediction_length']
        self.num_input_channels = checkpoint['num_input_channels']
        
        # Создаем модель и загружаем веса
        self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        print(f"📁 PatchTST модель загружена: {filepath}")


# Демо использования PatchTST
def demo_patchtst():
    """Демонстрация работы PatchTST модели"""
    print("🧠 ДЕМО: PatchTST модель")
    print("=" * 50)
    
    # Создаем синтетические данные для демо
    np.random.seed(42)
    
    # Параметры
    n_samples = 500
    context_length = 96  # 96 дней истории
    n_features = 12  # OHLCV + технические индикаторы + sentiment
    
    # Генерируем временные ряды с трендом
    X = []
    y = []
    
    for i in range(n_samples):
        # Создаем временной ряд с трендом и случайностью
        trend = np.linspace(100, 120, context_length)
        noise = np.random.randn(context_length, n_features) * 2
        base_series = trend.reshape(-1, 1) + noise
        
        # Добавляем цикличность
        time_idx = np.arange(context_length)
        seasonal = 5 * np.sin(2 * np.pi * time_idx / 20)
        base_series[:, 0] += seasonal  # Добавляем к первому признаку (close price)
        
        X.append(base_series)
        
        # Генерируем целевые переменные с некоторой зависимостью
        last_price = base_series[-1, 0]
        return_1d = np.random.randn() * 0.02 + np.tanh(base_series[-5:, 0].mean() - 110) * 0.01
        return_20d = return_1d * 1.5 + np.random.randn() * 0.03
        
        y.append([return_1d, return_20d])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Синтетические данные:")
    print(f"   X shape: {X.shape}")  
    print(f"   y shape: {y.shape}")
    
    # Разделяем на train/val/test
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Создаем и обучаем PatchTST модель
    model = ForecastPatchTST(
        context_length=context_length,
        prediction_length=20,  # Прогнозируем на 20 дней
        patch_length=12,  # Размер патча
        num_input_channels=n_features,
        d_model=64,  # Меньше для демо
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0.1
    )
    
    # Обучение (меньше эпох для демо)
    print(f"\n🚀 Начинаем обучение...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,  # Мало эпох для демо
        batch_size=16,
        learning_rate=1e-4
    )
    
    # Оценка
    print(f"\n📊 Оценка модели на test данных...")
    metrics = model.evaluate_model(X_test, y_test)
    
    print(f"\n📈 Метрики качества PatchTST:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Сравнение с простой базовой моделью
    baseline_mae_1d = np.mean(np.abs(y_test[:, 0]))  # Предсказываем 0
    baseline_mae_20d = np.mean(np.abs(y_test[:, 1]))
    
    improvement_1d = (1 - metrics['mae_1d'] / baseline_mae_1d) * 100
    improvement_20d = (1 - metrics['mae_20d'] / baseline_mae_20d) * 100
    
    print(f"\n🎯 Улучшение над baseline:")
    print(f"   1-day MAE: {improvement_1d:.1f}% лучше")
    print(f"   20-day MAE: {improvement_20d:.1f}% лучше")
    
    # Пример предсказаний
    predictions = model.predict(X_test[:3])
    print(f"\n🔮 Примеры предсказаний:")
    for i in range(3):
        print(f"   Пример {i+1}:")
        print(f"     True 1d: {y_test[i, 0]:.4f}, Pred: {predictions['return_1d'][i]:.4f}")
        print(f"     True 20d: {y_test[i, 1]:.4f}, Pred: {predictions['return_20d'][i]:.4f}")
        print(f"     Prob up 1d: {predictions['prob_1d'][i]:.3f}")
        print(f"     Prob up 20d: {predictions['prob_20d'][i]:.3f}")


# Unit-тест
def test_patchtst():
    """Unit-тест для PatchTST модели"""
    print("🧪 Unit-тест для PatchTST...")
    
    # Создаем маленькие тестовые данные
    X_test = np.random.randn(10, 24, 5)  # 10 примеров, 24 временных шага, 5 признаков
    y_test = np.random.randn(10, 2)      # 2 целевые переменные
    
    # Создаем модель
    model = ForecastPatchTST(
        context_length=24, 
        prediction_length=5,
        num_input_channels=5, 
        patch_length=6,
        d_model=32,
        num_hidden_layers=1
    )
    
    # Проверяем создание модели
    model._create_model()
    assert model.model is not None
    
    # Проверяем подготовку данных
    data_dict = model.prepare_data_for_patchtst(X_test, y_test)
    assert 'past_values' in data_dict
    assert data_dict['past_values'].shape == (10, 24, 5)
    
    print("✅ Unit-тест PatchTST пройден!")
