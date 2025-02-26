import asyncio
from typing import Dict, List, Any
import aiohttp
from concurrent.futures import ThreadPoolExecutor


class OracleClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.subscriptions = {}

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
        self.executor.shutdown()

    async def subscribe_to_feed(self, feed_id: str, callback):
        if feed_id not in self.subscriptions:
            self.subscriptions[feed_id] = []
        self.subscriptions[feed_id].append(callback)

    async def fetch_price_feed(self, asset: str) -> Dict[str, float]:
        endpoints = self.config['oracle_endpoints']
        tasks = []

        for endpoint in endpoints:
            task = asyncio.create_task(self._fetch_single_feed(endpoint, asset))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, dict)]

        return self._aggregate_price_data(valid_results)

    async def _fetch_single_feed(self, endpoint: str, asset: str) -> Dict[str, float]:
        try:
            async with self.session.get(f"{endpoint}/price/{asset}") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': float(data['price']),
                        'confidence': float(data['confidence']),
                        'timestamp': int(data['timestamp'])
                    }
        except Exception as e:
            print(f"Error fetching from {endpoint}: {str(e)}")
            return None

    def _aggregate_price_data(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return None

        total_confidence = sum(r['confidence'] for r in results)
        weighted_price = sum(r['price'] * r['confidence'] for r in results) / total_confidence

        return {
            'price': weighted_price,
            'confidence': total_confidence / len(results),
            'timestamp': max(r['timestamp'] for r in results)
        }