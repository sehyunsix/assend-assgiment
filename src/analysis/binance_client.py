import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Callable, Awaitable

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """
    WebSocket client for Binance Futures to stream required data.
    Streams: Trades, Orderbook, Liquidations, Ticker.
    """
    
    BASE_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.streams = [
            f"{self.symbol}@aggTrade",
            f"{self.symbol}@depth5@100ms",
            f"{self.symbol}@liquidationOrder",
            f"{self.symbol}@ticker"
        ]
        self.url = f"{self.BASE_URL}/{'/'.join(self.streams)}"
        self.running = False
        self.callbacks: List[Callable[[Dict], Awaitable[None]]] = []
        
    def add_callback(self, callback: Callable[[Dict], Awaitable[None]]):
        """Add an async callback for incoming messages."""
        self.callbacks.append(callback)

    async def start(self):
        """Start the WebSocket connection with automatic reconnection."""
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.url) as websocket:
                    logger.info(f"Connected to Binance WebSocket: {self.url}")
                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Add processing timestamp
                        data['_processing_ts'] = int(datetime.now().timestamp() * 1_000_000)
                        
                        # Distribute to callbacks
                        for callback in self.callbacks:
                            await callback(data)
                            
            except (websockets.ConnectionClosed, Exception) as e:
                if self.running:
                    logger.error(f"WebSocket connection lost: {e}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    logger.info("WebSocket client stopped.")

    def stop(self):
        """Stop the WebSocket connection."""
        self.running = False

async def example_callback(data: Dict):
    stream = data.get('e', data.get('stream'))
    print(f"Received data from stream: {stream}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = BinanceWebSocketClient()
    client.add_callback(example_callback)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(client.start())
    except KeyboardInterrupt:
        client.stop()
