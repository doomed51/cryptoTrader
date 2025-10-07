import ccxt 
from typing import Optional, Dict, Any, List, Tuple
import polars as pl
import pandas as pd
import logging
from decimal import Decimal, ROUND_DOWN
import time
from .position_calculator import PositionCalculator

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, exchange_id: str, api_key: str, secret: str, sandbox: bool = True, config_path: Optional[str] = None):
        """
        Initialize position manager with exchange connection
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: API key for exchange
            secret: API secret for exchange
            sandbox: Whether to use sandbox/testnet mode
            config_path: Path to strategy configuration files
        """
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.config_path = config_path
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        
        # Cache for positions and markets
        self._positions_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 30  # 30 seconds
        
        # Markets data
        self.markets = None
        self._load_markets()
        
        # Position calculator (initialized when needed)
        self._position_calculator = None
    
    def _load_markets(self):
        """Load market data from exchange"""
        try:
            self.markets = self.exchange.load_markets()
            logger.info(f"Loaded {len(self.markets)} markets from {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Fetch current account balance
        
        Returns:
            Dictionary of asset balances
        """
        try:
            balance = self.exchange.fetch_balance()
            return {
                asset: info['free'] + info['used'] 
                for asset, info in balance.items() 
                if isinstance(info, dict) and info.get('total', 0) > 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise
    
    def get_current_positions(self, strategy_id: Optional[str] = None) -> pl.DataFrame:
        """
        Fetch current positions from exchange
        
        Args:
            strategy_id: Optional strategy identifier for filtering
            
        Returns:
            Polars DataFrame with columns: symbol, quantity, market_value, weight
        """
        # Check cache
        current_time = time.time()
        if (current_time - self._cache_timestamp) < self._cache_ttl:
            positions_df = self._positions_cache.get('positions')
            if positions_df is not None:
                return self._filter_by_strategy(positions_df, strategy_id)
        
        try:
            # Fetch balance and prices
            balance = self.get_account_balance()
            prices = self._get_current_prices(list(balance.keys()))
            
            # Calculate positions
            positions_data = []
            total_value = 0
            
            for asset, quantity in balance.items():
                if quantity > 0:
                    price = prices.get(asset, 0)
                    market_value = quantity * price
                    total_value += market_value
                    
                    positions_data.append({
                        'symbol': asset,
                        'quantity': quantity,
                        'price': price,
                        'market_value': market_value,
                        'strategy_id': strategy_id or 'default'
                    })
            
            # Calculate weights
            for position in positions_data:
                position['weight'] = position['market_value'] / total_value if total_value > 0 else 0
            
            # Create DataFrame
            positions_df = pl.DataFrame(positions_data)
            
            # Update cache
            self._positions_cache['positions'] = positions_df
            self._cache_timestamp = current_time
            
            return self._filter_by_strategy(positions_df, strategy_id)
            
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise
    
    def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for given symbols
            Symbols should be in exchange expected format 
        """
        prices = {}
        try:
            # Create trading pairs (assume quote currency is USDT)
            tickers = {}
            for symbol in symbols:
                # if symbol == 'USDT' or symbol == 'USD':
                #     prices[symbol] = 1.0
                #     continue
                    
                # pair = f"{symbol}/USDT"
                if symbol in self.markets:
                    ticker = self.exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker['last']
                    
            return prices
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return {symbol: 0.0 for symbol in symbols}
    
    def get_position_calculator(self) -> PositionCalculator:
        """Get or create position calculator with current positions"""
        current_positions = self.get_current_positions()
        
        if self._position_calculator is None:
            self._position_calculator = PositionCalculator(current_positions, self.config_path)
        else:
            # Update with fresh positions
            self._position_calculator.update_current_positions(current_positions)
        
        return self._position_calculator

    def calculate_rebalancing_trades(self, 
                                   strategy_id: str,
                                   target_weights: Optional[pd.DataFrame] = None, 
                                   cash_allocation: Optional[float] = None) -> pl.DataFrame:
        """
        Calculate required trades to achieve target portfolio weights
        
        Args:
            strategy_id: Strategy identifier
            target_weights: Optional DataFrame with 'symbol' and 'weight' columns
            cash_allocation: Optional total cash to allocate for this strategy
            
        Returns:
            DataFrame with required trades
        """
        try:
            # Get current prices
            calculator = self.get_position_calculator()
            
            # Get all symbols that might be needed
            if target_weights is not None:
                symbols = target_weights['symbol'].tolist()
            else:
                config = calculator.get_strategy_config(strategy_id)
                symbols = list(config.get('target_weights', {}).keys())
            
            # Add any current position symbols
            current_positions = self.get_current_positions(strategy_id)
            if len(current_positions) > 0:
                symbols.extend(current_positions['symbol'].to_list())
            
            symbols = list(set(symbols))  # Remove duplicates
            prices = self._get_current_prices(symbols)
            
            # Use calculator to compute trades
            return calculator.calculate_rebalancing_trades(
                strategy_id=strategy_id,
                cash_allocation=cash_allocation,
                target_weights=target_weights,
                prices=prices
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate rebalancing trades: {e}")
            raise
    
    def execute_trades(self, trades_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        Execute calculated trades
        
        Args:
            trades_df: DataFrame with trade instructions
            
        Returns:
            List of execution results
        """
        results = []
        
        for trade in trades_df.iter_rows(named=True):
            try:
                symbol = trade['symbol']
                quantity = abs(trade['trade_quantity'])
                side = trade['side']
                
                # Create market symbol (assume USDT quote)
                market_symbol = f"{symbol}/USDT"
                
                if market_symbol not in self.markets:
                    logger.warning(f"Market {market_symbol} not available")
                    continue
                
                # Round quantity to exchange precision
                market_info = self.markets[market_symbol]
                precision = market_info.get('precision', {}).get('amount', 8)
                rounded_qty = float(Decimal(str(quantity)).quantize(
                    Decimal('0.' + '0' * (precision-1) + '1'), 
                    rounding=ROUND_DOWN
                ))
                
                if rounded_qty <= 0:
                    logger.warning(f"Quantity too small for {symbol}: {quantity}")
                    continue
                
                # Execute market order
                order = self.exchange.create_market_order(
                    market_symbol, 
                    side, 
                    rounded_qty
                )
                
                results.append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': rounded_qty,
                    'order_id': order['id'],
                    'status': order['status'],
                    'timestamp': order['timestamp']
                })
                
                logger.info(f"Executed {side} order for {rounded_qty} {symbol}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to execute trade for {trade['symbol']}: {e}")
                results.append({
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['trade_quantity'],
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def reconcile_positions(self, strategy_id: str = 'default') -> Dict[str, Any]:
        """
        Reconcile positions and detect discrepancies
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Reconciliation report
        """
        try:
            # Clear cache to get fresh data
            self._positions_cache.clear()
            
            # Get current positions
            positions = self.get_current_positions(strategy_id)
            
            # Calculate total portfolio value
            total_value = positions['market_value'].sum() if len(positions) > 0 else 0
            
            # Generate report
            report = {
                'strategy_id': strategy_id,
                'timestamp': time.time(),
                'total_positions': len(positions),
                'total_value': total_value,
                'positions': positions.to_dicts() if len(positions) > 0 else [],
                'status': 'success'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
            return {
                'strategy_id': strategy_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def _filter_by_strategy(self, df: pl.DataFrame, strategy_id: Optional[str]) -> pl.DataFrame:
        """Filter DataFrame by strategy ID if provided"""
        if strategy_id is None or len(df) == 0:
            return df
        
        if 'strategy_id' in df.columns:
            return df.filter(pl.col('strategy_id') == strategy_id)
        
        return df
    
    def analyze_portfolio_drift(self, strategy_id: str) -> Dict[str, Any]:
        """Analyze portfolio drift using position calculator"""
        calculator = self.get_position_calculator()
        return calculator.analyze_portfolio_drift(strategy_id)

    def get_portfolio_summary(self, strategy_id: str = 'default') -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        calculator = self.get_position_calculator()
        return calculator.get_portfolio_summary(strategy_id)
                'error': str(e),
                'status': 'failed'
            }
    
    def _filter_by_strategy(self, df: pl.DataFrame, strategy_id: Optional[str]) -> pl.DataFrame:
        """Filter DataFrame by strategy ID if provided"""
        if strategy_id is None or len(df) == 0:
            return df
        
        if 'strategy_id' in df.columns:
            return df.filter(pl.col('strategy_id') == strategy_id)
        
        return df
    
    def get_portfolio_summary(self, strategy_id: str = 'default') -> Dict[str, Any]:
        """
        Get portfolio summary for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Portfolio summary
        """
        try:
            positions = self.get_current_positions(strategy_id)
            
            if len(positions) == 0:
                return {
                    'strategy_id': strategy_id,
                    'total_value': 0,
                    'num_positions': 0,
                    'positions': []
                }
            
            return {
                'strategy_id': strategy_id,
                'total_value': positions['market_value'].sum(),
                'num_positions': len(positions),
                'largest_position': positions.sort('market_value', descending=True).head(1).to_dicts()[0],
                'positions': positions.to_dicts()
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {'error': str(e), 'strategy_id': strategy_id}

