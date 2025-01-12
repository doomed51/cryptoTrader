from enum import Enum
from typing import Optional
from dataclasses import dataclass

# Core Data Structures
class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None

@dataclass
class OrderResponse:
    order_id: str
    status: str
    filled_quantity: float
    average_price: float