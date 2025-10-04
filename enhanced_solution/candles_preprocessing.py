from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay


class DataPreprocessor:
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.scalers = {}
        
    def load_candles(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        assert not df.isnull().any().any(), "null values in Candles Data"
        assert len(df) > 0, "Empty Candles Data"

        df['begin'] = pd.to_datetime(df['begin'])
        df = df.sort_values(['ticker', 'begin'])

        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['begin'] = pd.to_datetime(df['begin'])
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            ticker_data['date_20bd_ago'] = ticker_data['begin'] - BDay(20)

            past_prices = ticker_data[['begin', 'close']].rename(
                columns={'begin': 'date_20bd_ago', 'close': 'past_close'}
            )
            
            ticker_data = ticker_data.merge(past_prices, on='date_20bd_ago', how='left')
            ticker_data['return_20d'] = (
                ticker_data['close'] - ticker_data['past_close']
            ) / ticker_data['past_close']
            
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()
            
            ticker_data['sma_5'] = ticker_data['close'].rolling(5).mean()
            ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
            
            delta = ticker_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            ticker_data['rsi'] = 100 - (100 / (1 + rs))
            
            ticker_data['volatility'] = (
                ticker_data['close'].pct_change().rolling(20).std()
            )
            

            df.loc[mask, 'sma_5'] = ticker_data['sma_5']
            df.loc[mask, 'sma_20'] = ticker_data['sma_20'] 
            df.loc[mask, 'rsi'] = ticker_data['rsi']
            df.loc[mask, 'volatility'] = ticker_data['volatility']
            
        return df
    
    def normalize_features(self, train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        
        for col in feature_cols:
            if col in train_df.columns:
                scaler = StandardScaler()
                train_values = train_df[col].dropna().values.reshape(-1, 1)
                scaler.fit(train_values)
                
                train_normalized[col] = scaler.transform(
                    train_df[col].values.reshape(-1, 1)
                ).flatten()

                test_normalized[col] = scaler.transform(
                    test_df[col].values.reshape(-1, 1)
                ).flatten()
                
                self.scalers[col] = scaler
        
        return train_normalized, test_normalized
    
    def create_windows(self, df: pd.DataFrame, 
                      feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.dropna()
            
            if len(ticker_data) < self.window_size + 20:
                continue
                
            for i in range(self.window_size, len(ticker_data) - 20):
                window_features = ticker_data[feature_cols].iloc[
                    i-self.window_size:i
                ].values
                
                target_1d = ticker_data['return_1d'].iloc[i]
                target_20d = ticker_data['return_20d'].iloc[i+19]
                
                if not (np.isnan(target_1d) or np.isnan(target_20d)):
                    X.append(window_features)
                    y.append([target_1d, target_20d])
        
        return np.array(X), np.array(y)