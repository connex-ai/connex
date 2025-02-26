import numpy as np
from typing import Dict, List, Any
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


class AnomalyDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'price_change',
            'volume_change',
            'volatility',
            'rsi',
            'macd',
            'price_deviation'
        ]
        self.detection_history = []

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame()

        # Price changes
        features['price_change'] = data['price'].pct_change()

        # Volume changes
        if 'volume' in data.columns:
            features['volume_change'] = data['volume'].pct_change()
        else:
            features['volume_change'] = 0

        # Volatility using rolling std
        features['volatility'] = data['price'].rolling(window=20).std()

        # RSI if available
        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        else:
            features['rsi'] = self._calculate_rsi(data['price'])

        # MACD if available
        if 'macd' in data.columns:
            features['macd'] = data['macd']
        else:
            features['macd'] = self._calculate_macd(data['price'])

        # Price deviation from moving average
        ma = data['price'].rolling(window=20).mean()
        features['price_deviation'] = (data['price'] - ma) / ma

        return features.fillna(0).values

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def train(self, training_data: pd.DataFrame):
        features = self.prepare_features(training_data)
        scaled_features = self.scaler.fit_transform(features)
        self.isolation_forest.fit(scaled_features)

    def detect(self, data: pd.DataFrame) -> Dict[str, Any]:
        features = self.prepare_features(data)
        scaled_features = self.scaler.transform(features)

        # Get anomaly scores (-1 for anomalies, 1 for normal)
        raw_scores = self.isolation_forest.score_samples(scaled_features)

        # Convert to probability-like scores (0 to 1)
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

        # Determine anomalies using dynamic thresholding
        threshold = self._calculate_dynamic_threshold(anomaly_scores)
        anomalies = anomaly_scores > threshold

        # Calculate feature contributions for anomalies
        feature_contributions = self._calculate_feature_contributions(scaled_features[anomalies])

        # Store detection results
        detection_result = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomalies_detected': anomalies.tolist(),
            'threshold': threshold,
            'feature_contributions': feature_contributions,
            'metadata': {
                'num_anomalies': int(np.sum(anomalies)),
                'average_score': float(np.mean(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores))
            }
        }

        self.detection_history.append(detection_result)
        return detection_result

    def _calculate_dynamic_threshold(self, scores: np.ndarray) -> float:
        # Use rolling statistics to determine threshold
        historical_mean = np.mean(scores)
        historical_std = np.std(scores)

        # Adjust threshold based on recent detection history
        if len(self.detection_history) > 0:
            recent_scores = [
                result['average_score']
                for result in self.detection_history[-10:]
            ]
            historical_mean = np.mean(recent_scores)
            historical_std = np.std(recent_scores)

        return historical_mean + (2 * historical_std)

    def _calculate_feature_contributions(self, anomalous_features: np.ndarray) -> Dict[str, float]:
        if len(anomalous_features) == 0:
            return {name: 0.0 for name in self.feature_names}

        # Calculate average deviation from mean for each feature
        feature_means = np.mean(anomalous_features, axis=0)
        feature_stds = np.std(anomalous_features, axis=0)

        # Calculate z-scores for feature contributions
        contributions = np.abs(feature_means / (feature_stds + 1e-10))

        # Normalize contributions
        contributions = contributions / np.sum(contributions)

        return dict(zip(self.feature_names, contributions.tolist()))

    def get_detection_stats(self) -> Dict[str, Any]:
        if not self.detection_history:
            return None

        recent_detections = self.detection_history[-100:]

        return {
            'total_detections': len(self.detection_history),
            'recent_anomaly_rate': np.mean([
                d['metadata']['num_anomalies'] > 0
                for d in recent_detections
            ]),
            'average_anomaly_score': np.mean([
                d['metadata']['average_score']
                for d in recent_detections
            ]),
            'feature_importance': self._calculate_cumulative_contributions(recent_detections),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_cumulative_contributions(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        cumulative = {name: 0.0 for name in self.feature_names}

        for detection in detections:
            contributions = detection['feature_contributions']
            for feature, contribution in contributions.items():
                cumulative[feature] += contribution

        # Normalize
        total = sum(cumulative.values())
        if total > 0:
            cumulative = {k: v / total for k, v in cumulative.items()}

        return cumulative

    def save_model(self, path: str):
        model_state = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'detection_history': self.detection_history,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_state, path)

    def load_model(self, path: str):
        model_state = joblib.load(path)
        self.isolation_forest = model_state['isolation_forest']
        self.scaler = model_state['scaler']
        self.detection_history = model_state['detection_history']