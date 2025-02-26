import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import jwt
from base64 import b64encode
import hashlib
import uuid


class SubscriptionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subscriptions = {}
        self.api_keys = {}
        self.usage_stats = {}
        self.jwt_secret = config.get('jwt_secret', 'your-secret-key')
        self.rate_limits = {
            'basic': 100,  # requests per minute
            'premium': 500,
            'enterprise': 2000
        }
        self.subscription_features = {
            'basic': {
                'historical_data_days': 30,
                'max_signals_per_day': 100,
                'custom_alerts': False,
                'api_access': True
            },
            'premium': {
                'historical_data_days': 180,
                'max_signals_per_day': 500,
                'custom_alerts': True,
                'api_access': True
            },
            'enterprise': {
                'historical_data_days': 365,
                'max_signals_per_day': 2000,
                'custom_alerts': True,
                'api_access': True
            }
        }

    async def create_subscription(self, user_id: str, plan: str, payment_info: Dict[str, Any]) -> Dict[str, Any]:
        if plan not in self.subscription_features:
            raise ValueError(f"Invalid subscription plan: {plan}")

        # Process payment (implement payment processing logic)
        payment_success = await self._process_payment(payment_info)

        if not payment_success:
            raise Exception("Payment processing failed")

        # Generate API credentials
        api_key = self._generate_api_key(user_id)
        api_secret = self._generate_api_secret()

        subscription = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'plan': plan,
            'features': self.subscription_features[plan],
            'api_credentials': {
                'key': api_key,
                'secret': api_secret
            },
            'status': 'active',
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=30)).isoformat(),
            'last_payment': datetime.now().isoformat()
        }

        self.subscriptions[subscription['id']] = subscription
        self.api_keys[api_key] = {
            'user_id': user_id,
            'subscription_id': subscription['id']
        }

        return subscription

    def _generate_api_key(self, user_id: str) -> str:
        timestamp = datetime.now().timestamp()
        random_component = uuid.uuid4().hex
        key_material = f"{user_id}:{timestamp}:{random_component}"
        return b64encode(hashlib.sha256(key_material.encode()).digest()).decode()

    def _generate_api_secret(self) -> str:
        return uuid.uuid4().hex

    async def _process_payment(self, payment_info: Dict[str, Any]) -> bool:
        # Implement payment processing logic
        return True

    def get_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        return self.subscriptions.get(subscription_id)

    def update_subscription(self, subscription_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if subscription_id not in self.subscriptions:
            raise ValueError(f"Subscription not found: {subscription_id}")

        subscription = self.subscriptions[subscription_id]

        # Update allowed fields
        allowed_updates = {'plan', 'status'}
        for field in allowed_updates:
            if field in updates:
                subscription[field] = updates[field]

        if 'plan' in updates:
            subscription['features'] = self.subscription_features[updates['plan']]

        return subscription

    async def validate_token(self, token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def generate_token(self, api_key: str, api_secret: str) -> str:
        if api_key not in self.api_keys:
            raise ValueError("Invalid API key")

        payload = {
            'api_key': api_key,
            'exp': datetime.utcnow() + timedelta(days=1)
        }

        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    async def check_rate_limit(self, api_key: str) -> bool:
        if api_key not in self.api_keys:
            return False

        subscription_id = self.api_keys[api_key]['subscription_id']
        subscription = self.subscriptions[subscription_id]

        # Initialize usage stats if not exists
        if api_key not in self.usage_stats:
            self.usage_stats[api_key] = {
                'requests': [],
                'daily_signals': 0,
                'last_reset': datetime.now()
            }

        # Clean up old requests
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        self.usage_stats[api_key]['requests'] = [
            req for req in self.usage_stats[api_key]['requests']
            if req > minute_ago
        ]

        # Check rate limit
        rate_limit = self.rate_limits[subscription['plan']]
        current_requests = len(self.usage_stats[api_key]['requests'])

        if current_requests >= rate_limit:
            return False

        # Add new request
        self.usage_stats[api_key]['requests'].append(now)
        return True

    async def track_signal_usage(self, api_key: str):
        if api_key not in self.usage_stats:
            return

        now = datetime.now()
        stats = self.usage_stats[api_key]

        # Reset daily counter if needed
        if now.date() > stats['last_reset'].date():
            stats['daily_signals'] = 0
            stats['last_reset'] = now

        stats['daily_signals'] += 1

    def get_usage_stats(self, api_key: str) -> Dict[str, Any]:
        if api_key not in self.usage_stats:
            return None

        stats = self.usage_stats[api_key]
        subscription_id = self.api_keys[api_key]['subscription_id']
        subscription = self.subscriptions[subscription_id]

        return {
            'requests_last_minute': len(stats['requests']),
            'signals_today': stats['daily_signals'],
            'rate_limit': self.rate_limits[subscription['plan']],
            'max_daily_signals': subscription['features']['max_signals_per_day']
        }

    async def cleanup_expired_subscriptions(self):
        now = datetime.now()
        expired_ids = []

        for sub_id, subscription in self.subscriptions.items():
            expires_at = datetime.fromisoformat(subscription['expires_at'])
            if expires_at < now:
                expired_ids.append(sub_id)

        for sub_id in expired_ids:
            subscription = self.subscriptions[sub_id]
            api_key = subscription['api_credentials']['key']

            del self.subscriptions[sub_id]
            if api_key in self.api_keys:
                del self.api_keys[api_key]
            if api_key in self.usage_stats:
                del self.usage_stats[api_key]