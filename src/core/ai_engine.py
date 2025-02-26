import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple
from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


class AIEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.sequence_length = config.get('sequence_length', 60)
        self.prediction_horizon = config.get('prediction_horizon', 12)
        self.initialize_models()

    def initialize_models(self):
        self.models['price_predictor'] = self._create_price_predictor()
        self.models['pattern_recognizer'] = self._create_pattern_recognizer()
        self.scalers['price'] = MinMaxScaler()

    def _create_price_predictor(self) -> Sequential:
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.prediction_horizon)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_pattern_recognizer(self) -> Sequential:
        model = Sequential([
            LSTM(256, input_shape=(self.sequence_length, 5), return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=False),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # Bearish, Neutral, Bullish
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_sequence_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        return np.array(X), np.array(y)

    def predict_prices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        sequence = self._prepare_price_data(input_data['price_history'])
        predictions = self.models['price_predictor'].predict(sequence)

        return {
            'predictions': self._inverse_transform_predictions(predictions),
            'confidence': self._calculate_prediction_confidence(predictions),
            'timestamp': datetime.now().isoformat()
        }

    def detect_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self._prepare_pattern_features(input_data)
        pattern_probabilities = self.models['pattern_recognizer'].predict(features)

        return {
            'pattern': self._interpret_pattern(pattern_probabilities),
            'probabilities': pattern_probabilities.tolist(),
            'confidence': float(np.max(pattern_probabilities)),
            'timestamp': datetime.now().isoformat()
        }

    def _prepare_price_data(self, price_history: List[float]) -> np.ndarray:
        scaled_data = self.scalers['price'].fit_transform(np.array(price_history).reshape(-1, 1))
        sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        return sequence

    def _prepare_pattern_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        features = np.column_stack([
            input_data['price_history'],
            input_data['volume_history'],
            input_data['rsi_history'],
            input_data['macd_history'],
            input_data['volatility_history']
        ])
        scaled_features = MinMaxScaler().fit_transform(features)
        return scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, 5)

    def _inverse_transform_predictions(self, predictions: np.ndarray) -> List[float]:
        return self.scalers['price'].inverse_transform(predictions).flatten().tolist()

    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> float:
        volatility = np.std(predictions)
        confidence = 1.0 / (1.0 + volatility)
        return float(confidence)

    def _interpret_pattern(self, probabilities: np.ndarray) -> str:
        pattern_index = np.argmax(probabilities)
        patterns = ['bearish', 'neutral', 'bullish']
        return patterns[pattern_index]

    def update_models(self, training_data: Dict[str, Any]):
        price_sequences, price_targets = self.prepare_sequence_data(training_data['price_history'])
        self.models['price_predictor'].fit(
            price_sequences,
            price_targets,
            epochs=self.config.get('training_epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            validation_split=0.2,
            verbose=0
        )

        self._save_model_state()

    def _save_model_state(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.models['price_predictor'].save(f'models/price_predictor_{timestamp}.h5')
        self.models['pattern_recognizer'].save(f'models/pattern_recognizer_{timestamp}.h5')
        joblib.dump(self.scalers, f'models/scalers_{timestamp}.pkl')