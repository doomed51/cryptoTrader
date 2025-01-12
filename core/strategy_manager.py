
from datetime import datetime
import logging 
import time
from core.order_manager import OrderManager
from base.order import OrderRequest, OrderSide, OrderType

def __init__(self, order_manager: OrderManager):
    self.order_manager = order_manager
    self.entry_time = None
    self.exit_time = None
    self.max_capital = None
    self.logger = logging.getLogger(__name__)

def set_strategy_params(self, entry_time: str, exit_time: str, max_capital: float):
    self.entry_time = datetime.strptime(entry_time, "%H:%M").time()
    self.exit_time = datetime.strptime(exit_time, "%H:%M").time()
    self.max_capital = max_capital

def run(self, symbol: str):
    while True:
        try:
            current_time = datetime.now().time()
            
            # Entry logic
            if current_time == self.entry_time:
                order = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=self._calculate_quantity(symbol)
                )
                self.order_manager.place_order(order)

            # Exit logic
            elif current_time == self.exit_time:
                order = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self._get_position_size(symbol)
                )
                self.order_manager.place_order(order)

            time.sleep(1)  # Sleep to prevent excessive CPU usage
            
        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
            break

def _calculate_quantity(self, symbol: str) -> float:
    account_info = self.order_manager.exchange.get_account_info()
    orderbook = self.order_manager.exchange.get_orderbook(symbol)
    current_price = float(orderbook['asks'][0][0])
    
    return min(
        self.max_capital / current_price,
        float(orderbook['asks'][0][1])  # Available liquidity
    )

def _get_position_size(self, symbol: str) -> float:
    account_info = self.order_manager.exchange.get_account_info()
    # Implement position size lookup
    return 0.0

def test_run(self, symbol: str, days: int = 1):
    # mock testing logic here: should print out order details at the expected times 
    pass