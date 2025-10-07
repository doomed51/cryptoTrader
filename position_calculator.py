import polars as pl
import pandas as pd
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

class PositionCalculator:
    def __init__(self, current_positions: pl.DataFrame, exchange, markets, config_path: Optional[str] = None):
        """
        Initialize position calculator with current positions
        
        Args:
            current_positions: DataFrame with current portfolio positions
            config_path: Optional path to strategy config directory
            exchange: CCXT exchange instance for fetching prices
        """
        self.current_positions = current_positions
        self.config_path = Path(config_path) if config_path else None
        self.strategy_configs = {}
        self.exchange = exchange
        self.markets = markets  # List of market symbols available on the exchange
        
        if self.config_path and self.config_path.exists():
            self._load_strategy_configs()
    
    def _load_strategy_configs(self):
        """Load all strategy configuration files from config directory"""
        try:
            config_files = list(self.config_path.glob("*.json")) + list(self.config_path.glob("*.yaml"))
            
            for config_file in config_files:
                strategy_name = config_file.stem
                
                if config_file.suffix == '.json':
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    continue
                
                self.strategy_configs[strategy_name] = config
                logger.info(f"Loaded strategy config: {strategy_name}")
                
        except Exception as e:
            logger.error(f"Failed to load strategy configs: {e}")
    
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
        

    def get_strategy_config(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy configuration dictionary
        """
        config = self.strategy_configs.get(strategy_id, {})
        
        # Default configuration structure
        default_config = {
            'name': strategy_id,
            'target_weights': {},
            'cash_allocation': 0,
            'rebalance_threshold': 0.05,  # 5% deviation threshold
            'min_trade_value': 10.0,
            'max_position_size': 0.20,  # 20% max position
            'allowed_assets': [],
            'risk_limits': {
                'max_drawdown': 0.10,
                'var_limit': 0.05
            }
        }
        
        # Merge with loaded config
        return {**default_config, **config}
    
    ################ how to load weights since some are in api, some are in local 
    def get_target_weights(self, 
                               strategy_id: str, 
                            #    custom_weights: Optional[pd.DataFrame] = None
                               ) -> pl.DataFrame:
        """
        Retrieve target weights for a strategy
        
        Args:
            strategy_id: Strategy identifier
            custom_weights: Optional custom weights DataFrame with 'symbol' and 'weight' columns
            
        Returns:
            DataFrame with target weights
        """
        try:
            if custom_weights is not None:
                # Use provided custom weights
                target_df = pl.from_pandas(custom_weights)
            else:
                # Load from strategy config
                config = self.get_strategy_config(strategy_id)
                target_weights = config.get('target_weights', {})
                
                if not target_weights:
                    raise ValueError(f"No target weights found for strategy {strategy_id}")
                
                # Convert to DataFrame
                weights_data = [
                    {'symbol': symbol, 'weight': weight}
                    for symbol, weight in target_weights.items()
                ]
                target_df = pl.DataFrame(weights_data)
            
            # Validate weights sum to 1.0 (allow small tolerance)
            total_weight = target_df['weight'].sum()
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Target weights sum to {total_weight}, normalizing to 1.0")
                target_df = target_df.with_columns([
                    (pl.col('weight') / total_weight).alias('weight')
                ])
            
            # Add strategy metadata
            target_df = target_df.with_columns([
                pl.lit(strategy_id).alias('strategy_id')
            ])
            
            return target_df
            
        except Exception as e:
            logger.error(f"Failed to calculate target weights for {strategy_id}: {e}")
            raise
    
    def calculate_rebalancing_trades(self, 
                                   strategy_id: str,
                                #    cash_allocation: Optional[float] = None,
                                #    target_weights: Optional[pd.DataFrame] = None,
                                #    prices: Optional[Dict[str, float]] = None
                                   ) -> pl.DataFrame:
        """
        Calculate required trades to achieve target portfolio weights
        
        Args:
            strategy_id: Strategy identifier
            cash_allocation: Total cash to allocate (overrides config)
            target_weights: Optional custom target weights
            prices: Current asset prices
            
        Returns:
            DataFrame with required trades
        """
        try:
            # Get strategy config
            config = self.get_strategy_config(strategy_id)
            
            # Use provided cash allocation or config default
            if cash_allocation is None:
                cash_allocation = config.get('cash_allocation', 0)
                target_leverage = config.get('target_leverage', 1)
                cash_allocation = cash_allocation * target_leverage

            if cash_allocation <= 0:
                raise ValueError(f"Invalid cash allocation: {cash_allocation}")
            
            # Calculate target weights
            target_df = self.get_target_weights(strategy_id)
            
            # Filter current positions for this strategy
            current_positions = self._filter_positions_by_strategy(strategy_id)
            
            # Get all symbols we need prices for
            all_symbols = set(target_df['symbol'].to_list())
            if len(current_positions) > 0:
                all_symbols.update(current_positions['symbol'].to_list())
            
            # Fetch current prices ---------------- SHOULD THIS INSTEAD BE THE ARRIVAL PRICES IN THE STARTEGY WEIGHTS ? 
            prices = self._get_current_prices(list(all_symbols))
            
            # Calculate target quantities
            target_quantities = []
            for row in target_df.iter_rows(named=True):
                symbol = row['symbol']
                weight = row['target_weight']
                price = prices.get(symbol, 0)
                
                if price <= 0:
                    logger.warning(f"No valid price for {symbol}, skipping")
                    continue
                
                target_value = cash_allocation * weight
                target_qty = target_value / price
                # ensure target_qty is rounded to the precision amount of the market 
                # https://github.com/ccxt/ccxt/wiki/Manual#precision-and-limits
                if symbol in self.markets:
                    precision = self.markets[symbol]['precision']['amount']
                    target_qty = float(Decimal(target_qty).quantize(Decimal(str(precision)), rounding=ROUND_DOWN))
                    target_value = target_qty * price
                
                target_quantities.append({
                    'symbol': symbol,
                    'target_weight': weight,
                    'target_quantity': target_qty,
                    'target_value': target_value,
                    'price': price
                })
            
            if not target_quantities:
                raise ValueError("No valid target quantities calculated")
            
            target_qty_df = pl.DataFrame(target_quantities)
            
            # Merge with current positions
            if len(current_positions) > 0:
                trades_df = target_qty_df.join(
                    current_positions.select(['symbol', 'quantity']),
                    on='symbol',
                    how='outer'
                ).fill_null(0)
            else:
                trades_df = target_qty_df.with_columns([
                    pl.lit(0.0).alias('quantity')
                ])
            
            # Calculate trade quantities and values
            trades_df = trades_df.with_columns([
                (pl.col('target_quantity') - pl.col('quantity')).alias('trade_quantity'),
                ((pl.col('target_quantity') - pl.col('quantity')) * pl.col('price')).alias('trade_value'),
                pl.lit(strategy_id).alias('strategy_id')
            ])
            
            # Apply filters based on strategy config
            min_trade_value = config.get('min_trade_value', 10.0)
            rebalance_threshold = config.get('rebalance_threshold', 0.05)
            
            # Filter out small trades
            trades_df = trades_df.filter(
                pl.col('trade_value').abs() > min_trade_value
            )
            
            # Filter based on rebalance threshold (% of portfolio)
            if len(trades_df) > 0:
                trades_df = trades_df.filter(
                    (pl.col('trade_value').abs() / cash_allocation) > rebalance_threshold
                )
            
            # Add trade direction and metadata
            # if len(trades_df) > 0:
            #     trades_df = trades_df.with_columns([
            #         pl.when(pl.col('trade_quantity') > 0) # trade quantity should imply direction, don't need a side column 
            #         .then(pl.lit('buy'))
            #         .otherwise(pl.lit('sell'))
            #         .alias('side'),
            #         pl.col('trade_quantity').abs().alias('abs_quantity'),
            #         (pl.col('target_value') / cash_allocation).alias('target_weight_pct')
            #     ])
            
            return trades_df
            
        except Exception as e:
            logger.error(f"Failed to calculate rebalancing trades: {e}")
            raise
    
    def analyze_portfolio_drift(self, 
                              strategy_id: str,
                              target_weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze how current portfolio has drifted from target weights
        
        Args:
            strategy_id: Strategy identifier
            target_weights: Optional target weights (uses config if not provided)
            
        Returns:
            Portfolio drift analysis
        """
        try:
            # Get target weights
            target_df = self.get_target_weights(strategy_id, target_weights)
            
            # Get current positions for strategy
            current_positions = self._filter_positions_by_strategy(strategy_id)
            
            if len(current_positions) == 0:
                return {
                    'strategy_id': strategy_id,
                    'drift_analysis': 'No current positions found',
                    'total_drift': 0,
                    'positions_analysis': []
                }
            
            # Calculate total portfolio value
            total_value = current_positions['market_value'].sum()
            
            # Merge target and current weights
            analysis_df = target_df.join(
                current_positions.select(['symbol', 'market_value', 'weight']),
                on='symbol',
                how='outer'
            ).fill_null(0)
            
            # Calculate drift metrics
            analysis_df = analysis_df.with_columns([
                (pl.col('weight') - pl.col('target_weight')).alias('weight_drift'),
                ((pl.col('weight') - pl.col('target_weight')).abs()).alias('abs_drift'),
                (pl.col('market_value') / total_value * 100).alias('current_weight_pct'),
                (pl.col('target_weight') * 100).alias('target_weight_pct')
            ])
            
            # Calculate summary metrics
            total_drift = analysis_df['abs_drift'].sum()
            max_drift = analysis_df['abs_drift'].max()
            
            positions_analysis = []
            for row in analysis_df.iter_rows(named=True):
                positions_analysis.append({
                    'symbol': row['symbol'],
                    'current_weight': row['weight'],
                    'target_weight': row['target_weight'],
                    'drift': row['weight_drift'],
                    'abs_drift': row['abs_drift'],
                    'market_value': row['market_value'],
                    'needs_rebalancing': abs(row['weight_drift']) > 0.05
                })
            
            return {
                'strategy_id': strategy_id,
                'total_portfolio_value': total_value,
                'total_drift': total_drift,
                'max_drift': max_drift,
                'avg_drift': total_drift / len(analysis_df) if len(analysis_df) > 0 else 0,
                'positions_needing_rebalance': sum(1 for p in positions_analysis if p['needs_rebalancing']),
                'positions_analysis': positions_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze portfolio drift: {e}")
            return {'error': str(e), 'strategy_id': strategy_id}
    
    def validate_trade_constraints(self, 
                                 trades_df: pl.DataFrame, 
                                 strategy_id: str) -> pl.DataFrame:
        """
        Validate trades against strategy constraints and risk limits
        
        Args:
            trades_df: DataFrame with calculated trades
            strategy_id: Strategy identifier
            
        Returns:
            Validated trades DataFrame with constraint checks
        """
        try:
            if len(trades_df) == 0:
                return trades_df
            
            config = self.get_strategy_config(strategy_id)
            max_position_size = config.get('max_position_size', 0.20)
            allowed_assets = config.get('allowed_assets', [])
            
            # Add validation columns
            validated_df = trades_df.with_columns([
                pl.lit(True).alias('valid_trade'),
                pl.lit('').alias('validation_errors')
            ])
            
            # Check position size limits
            validated_df = validated_df.with_columns([
                pl.when(pl.col('target_weight_pct') > max_position_size)
                .then(pl.lit(False))
                .otherwise(pl.col('valid_trade'))
                .alias('valid_trade'),
                
                pl.when(pl.col('target_weight_pct') > max_position_size)
                .then(pl.col('validation_errors') + f"Position size exceeds {max_position_size*100}%; ")
                .otherwise(pl.col('validation_errors'))
                .alias('validation_errors')
            ])
            
            # Check allowed assets (if specified)
            if allowed_assets:
                validated_df = validated_df.with_columns([
                    pl.when(~pl.col('symbol').is_in(allowed_assets))
                    .then(pl.lit(False))
                    .otherwise(pl.col('valid_trade'))
                    .alias('valid_trade'),
                    
                    pl.when(~pl.col('symbol').is_in(allowed_assets))
                    .then(pl.col('validation_errors') + "Asset not in allowed list; ")
                    .otherwise(pl.col('validation_errors'))
                    .alias('validation_errors')
                ])
            
            return validated_df
            
        except Exception as e:
            logger.error(f"Failed to validate trade constraints: {e}")
            return trades_df
    
    def _filter_positions_by_strategy(self, strategy_id: str) -> pl.DataFrame:
        """Filter current positions by strategy ID"""
        if len(self.current_positions) == 0:
            return pl.DataFrame()
        
        if 'strategy_id' in self.current_positions.columns:
            return self.current_positions.filter(
                pl.col('strategy_id') == strategy_id
            )
        
        # If no strategy_id column, return all positions (assume single strategy)
        return self.current_positions
    
    def update_current_positions(self, new_positions: pl.DataFrame):
        """Update the current positions DataFrame"""
        self.current_positions = new_positions
    
    def get_portfolio_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Portfolio summary with positions, allocations, and metrics
        """
        try:
            positions = self._filter_positions_by_strategy(strategy_id)
            config = self.get_strategy_config(strategy_id)
            
            if len(positions) == 0:
                return {
                    'strategy_id': strategy_id,
                    'status': 'No positions',
                    'total_value': 0,
                    'num_positions': 0,
                    'config': config
                }
            
            total_value = positions['market_value'].sum()
            
            # Calculate concentration metrics
            position_weights = positions['weight'].to_list()
            max_weight = max(position_weights) if position_weights else 0
            
            # Get top positions
            top_positions = positions.sort('market_value', descending=True).head(5)
            
            return {
                'strategy_id': strategy_id,
                'total_value': total_value,
                'num_positions': len(positions),
                'max_position_weight': max_weight,
                'concentration_ratio': sum(sorted(position_weights, reverse=True)[:3]),  # Top 3 positions
                'top_positions': top_positions.to_dicts(),
                'all_positions': positions.to_dicts(),
                'config_summary': {
                    'cash_allocation': config.get('cash_allocation', 0),
                    'rebalance_threshold': config.get('rebalance_threshold', 0.05),
                    'max_position_size': config.get('max_position_size', 0.20)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {'error': str(e), 'strategy_id': strategy_id}
