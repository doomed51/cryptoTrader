# Exchange Interface
from abc import ABC, abstractmethod
from typing import List, Dict
from base.order import OrderRequest, OrderResponse

class ExchangeInterface(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        pass
    
    @abstractmethod
    def post_order(self, order: OrderRequest) -> OrderResponse:
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        pass
    
    @abstractmethod
    def get_orderbook(self, symbol: str) -> Dict:
        pass

# Example Exchange Implementation
class BinanceExchange(ExchangeInterface):
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        # Initialize actual exchange client here

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        # Implement actual exchange API call
        pass

    def post_order(self, order: OrderRequest) -> OrderResponse:
        # Implement actual exchange API call
        pass

    def get_account_info(self) -> Dict:
        # Implement actual exchange API call
        pass

    def get_orderbook(self, symbol: str) -> Dict:
        # Implement actual exchange API call
        pass