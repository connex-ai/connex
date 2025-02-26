import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.buffer_size = config.get('buffer_size', 1000)
        self.data_buffer = []

    def process_raw_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        processed_data = self._preprocess_data(data)
        filtered_data = self._apply_filters(processed_data)
        enriched_data = self._enrich_data(filtered_data)
        return enriched_data

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if len(self.data_buffer) >= self.buffer_size:
            self.data_buffer.pop(0)

        self.data_buffer.append(data)

        df = pd.DataFrame(self.data_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        return {
            'raw_data': data,
            'dataframe': df,
            'processed_at': datetime.now().isoformat()
        }

    def _apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']

        # Kalman filter for noise reduction
        df['filtered_price'] = self._apply_kalman_filter(df['price'])

        # Moving average convergence divergence
        df['macd'] = self._calculate_macd(df['filtered_price'])

        # Relative strength index
        df['rsi'] = self._calculate_rsi(df['filtered_price'])

        data['dataframe'] = df
        return data

    def _enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']

        # Calculate volatility
        df['volatility'] = df['filtered_price'].rolling(window=20).std()

        # Calculate price momentum
        df['momentum'] = df['filtered_price'].diff(periods=5)

        # Calculate trading volume intensity
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_intensity'] = df['volume'] / df['volume_ma']

        data['dataframe'] = df
        return data

    def _apply_kalman_filter(self, series: pd.Series) -> pd.Series:
        n = len(series)
        filtered_series = np.zeros(n)

        # Kalman filter parameters
        Q = 1e-5  # Process variance
        R = 0.1  # Measurement variance

        filtered_series[0] = series.iloc[0]
        P = 1.0

        for i in range(1, n):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            filtered_series[i] = filtered_series[i - 1] + K * (series.iloc[i] - filtered_series[i - 1])
            P = (1 - K) * P_pred

        return pd.Series(filtered_series, index=series.index)

    def _calculate_macd(self, series: pd.Series) -> pd.Series:
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd

    def _calculate_rsi(self, series: pd.Series, periods: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))