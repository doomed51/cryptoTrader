
from risk_manager import RiskManager
from exchange_manager import ExchangeInterface
from enum import Enum
from typing import Dict, List, Optional
import logging
from base.order import OrderRequest, OrderResponse, OrderSide, OrderType

# Order Management
class OrderManager:
    def __init__(self, exchange: ExchangeInterface, risk_manager: RiskManager):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.active_orders: Dict[str, OrderRequest] = {}
        self.logger = logging.getLogger(__name__)

    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        # Check for existing orders
        if self._has_existing_orders(order.symbol):
            self.logger.warning(f"Existing orders found for {order.symbol}")
            return None

        # Get account information and check risk
        account_info = self.exchange.get_account_info()
        if not self.risk_manager.check_order_risk(order, account_info):
            self.logger.warning("Order failed risk checks")
            return None

        # Handle illiquid markets
        if order.order_type == OrderType.MARKET:
            orderbook = self.exchange.get_orderbook(order.symbol)
            order = self._adjust_for_liquidity(order, orderbook)

        # Place the main order
        try:
            response = self.exchange.post_order(order)
            self.active_orders[response.order_id] = order
            
            # Place stop loss if entry order
            if order.side == OrderSide.BUY:
                self._place_stop_loss(order, response.average_price)
                
            return response
        except Exception as e:
            self.logger.error(f"Order placement failed: {str(e)}")
            return None

    def _has_existing_orders(self, symbol: str) -> bool:
        return any(order.symbol == symbol for order in self.active_orders.values())

    def _adjust_for_liquidity(self, order: OrderRequest, orderbook: Dict) -> OrderRequest:
        # Implement smart order routing logic for illiquid markets
        return order

    def _place_stop_loss(self, entry_order: OrderRequest, entry_price: float):
        sl_price = entry_price * 0.95  # 5% stop loss
        sl_order = OrderRequest(
            symbol=entry_order.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=entry_order.quantity,
            stop_price=sl_price
        )
        self.exchange.post_order(sl_order)

