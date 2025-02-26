import asyncio
import json
import websockets
from typing import Dict, Any, Callable, Optional
from datetime import datetime


class PythOracle:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ws = None
        self.subscribers = {}
        self.connection_retry_delay = 1
        self.max_retry_delay = 60
        self.connected = False
        self.last_heartbeat = None

    async def connect(self):
        while True:
            try:
                self.ws = await websockets.connect(
                    self.config['pyth_ws_url'],
                    extra_headers={'X-API-Key': self.config['pyth_api_key']}
                )
                self.connected = True
                self.connection_retry_delay = 1
                await self._handle_connection()
            except Exception as e:
                print(f"Pyth connection error: {str(e)}")
                self.connected = False
                await asyncio.sleep(self.connection_retry_delay)
                self.connection_retry_delay = min(self.connection_retry_delay * 2, self.max_retry_delay)

    async def _handle_connection(self):
        try:
            await self._subscribe_to_feeds()
            while True:
                message = await self.ws.recv()
                await self._process_message(json.loads(message))
        except websockets.ConnectionClosed:
            self.connected = False

    async def _subscribe_to_feeds(self):
        subscription_message = {
            "type": "subscribe",
            "products": self.config['pyth_products']
        }
        await self.ws.send(json.dumps(subscription_message))

    async def _process_message(self, message: Dict[str, Any]):
        if message['type'] == 'price_update':
            product_id = message['product_id']
            if product_id in self.subscribers:
                price_data = self._format_price_data(message)
                for callback in self.subscribers[product_id]:
                    await callback(price_data)
        elif message['type'] == 'heartbeat':
            self.last_heartbeat = datetime.now()

    def _format_price_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'source': 'pyth',
            'product_id': message['product_id'],
            'price': float(message['price']),
            'confidence': float(message['confidence']),
            'timestamp': int(message['timestamp']),
            'status': message.get('trading_status', 'unknown'),
            'metadata': {
                'ema_price': float(message.get('ema_price', 0)),
                'volume': float(message.get('volume', 0)),
                'slot': int(message.get('slot', 0))
            }
        }

    async def subscribe(self, product_id: str, callback: Callable):
        if product_id not in self.subscribers:
            self.subscribers[product_id] = set()
        self.subscribers[product_id].add(callback)

        if self.connected:
            subscription_message = {
                "type": "subscribe",
                "products": [product_id]
            }
            await self.ws.send(json.dumps(subscription_message))

    async def unsubscribe(self, product_id: str, callback: Optional[Callable] = None):
        if product_id in self.subscribers:
            if callback:
                self.subscribers[product_id].remove(callback)
                if not self.subscribers[product_id]:
                    del self.subscribers[product_id]
            else:
                del self.subscribers[product_id]

        if self.connected and product_id not in self.subscribers:
            unsubscribe_message = {
                "type": "unsubscribe",
                "products": [product_id]
            }
            await self.ws.send(json.dumps(unsubscribe_message))

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.connected = False