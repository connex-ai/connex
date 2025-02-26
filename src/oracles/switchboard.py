import asyncio
import json
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import aiohttp
from asyncio import Queue
from base58 import b58encode, b58decode


class SwitchboardOracle:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.aggregator_queues = {}
        self.subscribers = {}
        self.update_interval = config.get('update_interval', 1.0)
        self.metrics = {
            'updates_processed': 0,
            'errors_encountered': 0,
            'latest_latency': 0.0
        }

    async def initialize(self):
        self.session = aiohttp.ClientSession()
        for aggregator_pubkey in self.config['aggregator_pubkeys']:
            self.aggregator_queues[aggregator_pubkey] = Queue()
            asyncio.create_task(self._process_aggregator_queue(aggregator_pubkey))

    async def _process_aggregator_queue(self, aggregator_pubkey: str):
        queue = self.aggregator_queues[aggregator_pubkey]
        while True:
            try:
                start_time = datetime.now()
                result = await self._fetch_aggregator_result(aggregator_pubkey)

                if result:
                    self.metrics['updates_processed'] += 1
                    self.metrics['latest_latency'] = (datetime.now() - start_time).total_seconds()

                    await queue.put(result)

                    if aggregator_pubkey in self.subscribers:
                        formatted_data = self._format_aggregator_data(result)
                        for callback in self.subscribers[aggregator_pubkey]:
                            await callback(formatted_data)

            except Exception as e:
                self.metrics['errors_encountered'] += 1
                print(f"Error processing aggregator {aggregator_pubkey}: {str(e)}")

            await asyncio.sleep(self.update_interval)

    async def _fetch_aggregator_result(self, aggregator_pubkey: str) -> Dict[str, Any]:
        try:
            url = f"{self.config['switchboard_rpc_url']}/aggregator/{aggregator_pubkey}/latest"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching aggregator data: {response.status}")
                    return None
        except Exception as e:
            print(f"Exception in fetch_aggregator_result: {str(e)}")
            return None

    def _format_aggregator_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'source': 'switchboard',
            'aggregator_pubkey': result['aggregatorPubkey'],
            'value': float(result['result']),
            'decimal_places': int(result.get('decimalPlaces', 0)),
            'confidence_interval': float(result.get('confidenceInterval', 0)),
            'min_response': float(result.get('minResponse', 0)),
            'max_response': float(result.get('maxResponse', 0)),
            'timestamp': int(result['timestamp']),
            'num_success': int(result.get('numSuccess', 0)),
            'num_error': int(result.get('numError', 0)),
            'round_id': int(result.get('roundId', 0))
        }

    async def subscribe(self, aggregator_pubkey: str, callback: Callable):
        if aggregator_pubkey not in self.subscribers:
            self.subscribers[aggregator_pubkey] = set()
            if aggregator_pubkey not in self.aggregator_queues:
                self.aggregator_queues[aggregator_pubkey] = Queue()
                asyncio.create_task(self._process_aggregator_queue(aggregator_pubkey))

        self.subscribers[aggregator_pubkey].add(callback)

    async def unsubscribe(self, aggregator_pubkey: str, callback: Optional[Callable] = None):
        if aggregator_pubkey in self.subscribers:
            if callback:
                self.subscribers[aggregator_pubkey].remove(callback)
                if not self.subscribers[aggregator_pubkey]:
                    del self.subscribers[aggregator_pubkey]
                    if aggregator_pubkey in self.aggregator_queues:
                        del self.aggregator_queues[aggregator_pubkey]
            else:
                del self.subscribers[aggregator_pubkey]
                if aggregator_pubkey in self.aggregator_queues:
                    del self.aggregator_queues[aggregator_pubkey]

    async def get_latest_value(self, aggregator_pubkey: str) -> Dict[str, Any]:
        if aggregator_pubkey not in self.aggregator_queues:
            return None

        queue = self.aggregator_queues[aggregator_pubkey]
        if queue.empty():
            result = await self._fetch_aggregator_result(aggregator_pubkey)
            if result:
                await queue.put(result)
                return self._format_aggregator_data(result)
            return None

        return self._format_aggregator_data(queue.get_nowait())

    async def get_metrics(self) -> Dict[str, Any]:
        return {
            **self.metrics,
            'active_aggregators': len(self.aggregator_queues),
            'total_subscribers': sum(len(subs) for subs in self.subscribers.values()),
            'queue_sizes': {
                pubkey: queue.qsize()
                for pubkey, queue in self.aggregator_queues.items()
            }
        }

    async def close(self):
        if self.session:
            await self.session.close()