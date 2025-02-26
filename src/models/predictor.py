import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from bayes_opt import BayesianOptimization
import joblib
import pandas as pd
from datetime import datetime, timedelta


class PricePredictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.optimization_results = {}
        self.feature_importance = {}
        self.cross_val_scores = {}
        self.initialize_models()

    def initialize_models(self):
        self.models['gbm'] = self._create_gbm_model()
        self.models['rf'] = self._create_rf_model()

    def _create_gbm_model(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def _create_rf_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        def gbm_objective(learning_rate, max_depth, n_estimators):
            params = {
                'learning_rate': learning_rate,
                'max_depth': int(max_depth),
                'n_estimators': int(n_estimators)
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            return self._evaluate_model(model, X, y)

        optimizer = BayesianOptimization(
            f=gbm_objective,
            pbounds={
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'n_estimators': (50, 300)
            },
            random_state=42
        )

        optimizer.maximize(init_points=5, n_iter=20)
        self.optimization_results['gbm'] = optimizer.max

        # Update model with optimal parameters
        best_params = optimizer.max['params']
        self.models['gbm'] = GradientBoostingRegressor(
            learning_rate=best_params['learning_rate'],
            max_depth=int(best_params['max_depth']),
            n_estimators=int(best_params['n_estimators']),
            random_state=42
        )

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = -mean_squared_error(y_val, y_pred, squared=False)
            scores.append(score)

        return np.mean(scores)

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'volatility',
            'price_momentum', 'volume_intensity'
        ]

        X = data[feature_columns].values
        y = data['price'].shift(-1).values[:-1]  # Next period's price

        return X[:-1], y

    def train(self, training_data: pd.DataFrame):
        X, y = self.prepare_features(training_data)

        # Optimize hyperparameters if enabled
        if self.config.get('optimize_hyperparameters', False):
            self.optimize_hyperparameters(X, y)

        # Train models
        for name, model in self.models.items():
            model.fit(X, y)
            self.feature_importance[name] = dict(zip(
                training_data.columns,
                model.feature_importances_
            ))

        self._evaluate_models(X, y)

    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            scores = {
                'rmse': [],
                'mape': []
            }

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                scores['rmse'].append(mean_squared_error(y_val, y_pred, squared=False))
                scores['mape'].append(mean_absolute_percentage_error(y_val, y_pred))

            self.cross_val_scores[name] = {
                'rmse_mean': np.mean(scores['rmse']),
                'rmse_std': np.std(scores['rmse']),
                'mape_mean': np.mean(scores['mape']),
                'mape_std': np.std(scores['mape'])
            }

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(features)
            pred_std = self._estimate_prediction_uncertainty(model, features)