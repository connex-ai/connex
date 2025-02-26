import os
import yaml
from typing import Dict, Any, Optional
import json
from pathlib import Path


class Config:
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load_config()

    def load_config(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.getenv('CONNEX_CONFIG_PATH', 'config/default.yaml')

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            self._config = self._load_default_config()

        # Override with environment variables
        self._override_from_env()

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            'app': {
                'name': 'Connex Oracle',
                'version': '1.0.0',
                'environment': 'development'
            },
            'oracle': {
                'providers': ['pyth', 'switchboard'],
                'update_interval': 1.0,
                'cache_duration': 300,
                'max_retries': 3
            },
            'ai': {
                'model_path': 'models/',
                'training_interval': 3600,
                'prediction_horizon': 12,
                'sequence_length': 60,
                'min_confidence': 0.7
            },
            'network': {
                'rpc_url': 'https://api.mainnet-beta.solana.com',
                'ws_url': 'wss://api.mainnet-beta.solana.com',
                'commitment': 'confirmed'
            },
            'security': {
                'jwt_secret': os.getenv('CONNEX_JWT_SECRET', 'default-secret-key'),
                'api_rate_limit': 100,
                'max_connections': 1000
            },
            'subscription': {
                'plans': {
                    'basic': {
                        'price': 50,
                        'rate_limit': 100,
                        'features': ['basic_signals', 'api_access']
                    },
                    'premium': {
                        'price': 200,
                        'rate_limit': 500,
                        'features': ['advanced_signals', 'custom_alerts', 'api_access']
                    },
                    'enterprise': {
                        'price': 1000,
                        'rate_limit': 2000,
                        'features': ['all']
                    }
                }
            },
            'database': {
                'url': os.getenv('CONNEX_DB_URL', 'postgresql://localhost/connex'),
                'pool_size': 20,
                'max_overflow': 10
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/connex.log'
            }
        }

    def _override_from_env(self):
        env_prefix = 'CONNEX_'
        for key in os.environ:
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                self._set_nested_config(config_key.split('_'), os.environ[key])

    def _set_nested_config(self, keys: list, value: str):
        current = self._config
        for key in keys[:-1]:
            current = current.setdefault(key, {})

        try:
            # Try to parse as JSON first
            parsed_value = json.loads(value)
            current[keys[-1]] = parsed_value
        except json.JSONDecodeError:
            # If not valid JSON, store as string
            current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    def save(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.getenv('CONNEX_CONFIG_PATH', 'config/default.yaml')

        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

    def reset(self):
        self._config = self._load_default_config()
        self._override_from_env()

    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()

    def update(self, updates: Dict[str, Any]):
        def deep_update(source, updates):
            for key, value in updates.items():
                if key in source and isinstance(source[key], dict) and isinstance(value, dict):
                    deep_update(source[key], value)
                else:
                    source[key] = value

        deep_update(self._config, updates)