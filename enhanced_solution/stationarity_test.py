from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class StationarityTester:
    """Класс для проведения тестов стационарности"""
    
    @staticmethod
    def adf_test(series: pd.Series, significance_level: float = 0.05) -> Dict:
        """
        Выполнение ADF-теста на стационарность
        
        Returns:
            dict: результаты теста с интерпретацией
        """
        
        if len(series) < 10:
            return {
                'statistic': None,
                'p_value': None, 
                'is_stationary': False,
                'error': 'Недостаточно данных для теста'
            }
        
        try:
            # Выполняем ADF тест
            result = adfuller(series, autolag='AIC')
            
            adf_statistic = result
            p_value = result
            critical_values = result
            
            # Интерпретация результата
            is_stationary = p_value <= significance_level
            
            return {
                'statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'interpretation': (
                    f"Ряд {'стационарен' if is_stationary else 'нестационарен'} "
                    f"(p-value: {p_value:.4f})"
                ),
                'error': None
            }
            
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'is_stationary': False,
                'error': f"Ошибка при выполнении ADF теста: {str(e)}"
            }
    
    def test_multiple_series(self, df: pd.DataFrame, 
                           target_column: str = 'close') -> pd.DataFrame:
        """Тестирование стационарности для множества временных рядов"""
        
        results = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            
            # Тест уровней цен
            levels_result = self.adf_test(ticker_data[target_column])
            
            # Тест первых разностей (доходностей)
            returns = ticker_data[target_column].pct_change()
            returns_result = self.adf_test(returns)
            
            results.append({
                'ticker': ticker,
                'levels_stationary': levels_result['is_stationary'],
                'levels_p_value': levels_result['p_value'],
                'returns_stationary': returns_result['is_stationary'], 
                'returns_p_value': returns_result['p_value'],
                'recommendation': self._get_recommendation(
                    levels_result['is_stationary'],
                    returns_result['is_stationary']
                )
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _get_recommendation(levels_stat: bool, returns_stat: bool) -> str:
        """Рекомендации по обработке ряда"""
        if levels_stat:
            return "Использовать уровни цен"
        elif returns_stat:
            return "Использовать доходности (разности)"
        else:
            return "Требуется дополнительная обработка"
    
    def make_stationary(self, series: pd.Series, 
                       method: str = 'diff') -> Tuple[pd.Series, str]:
        """
        Приведение ряда к стационарному виду
        
        Args:
            series: исходный ряд
            method: метод ('diff', 'log_diff', 'detrend')
        """
        if method == 'diff':
            stationary_series = series.diff().dropna()
            description = "Первые разности"
            
        elif method == 'log_diff': 
            log_series = np.log(series.replace(0, np.nan))
            stationary_series = log_series.diff().dropna()
            description = "Логарифмические доходности"
            
        elif method == 'detrend':
            # Простое удаление линейного тренда
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)
            stationary_series = series - trend
            description = "Удаление линейного тренда"
            
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        return stationary_series, description