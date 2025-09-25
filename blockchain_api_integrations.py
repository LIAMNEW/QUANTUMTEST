import requests
import json
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinAPIClient:
    """Bitcoin blockchain node API client"""
    
    def __init__(self, node_url: Optional[str] = None, api_key: Optional[str] = None):
        self.node_url = node_url or os.getenv('BITCOIN_NODE_URL', 'https://blockstream.info/api')
        self.api_key = api_key or os.getenv('BITCOIN_API_KEY')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
    
    def get_transaction(self, txid: str) -> Dict[str, Any]:
        """Get Bitcoin transaction details by transaction ID"""
        try:
            response = self.session.get(f"{self.node_url}/tx/{txid}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Bitcoin transaction {txid}: {e}")
            return {}
    
    def get_address_transactions(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions for a Bitcoin address"""
        try:
            response = self.session.get(f"{self.node_url}/address/{address}/txs")
            response.raise_for_status()
            transactions = response.json()
            return transactions[:limit] if len(transactions) > limit else transactions
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Bitcoin address transactions for {address}: {e}")
            return []
    
    def get_block_transactions(self, block_hash: str) -> List[Dict[str, Any]]:
        """Get all transactions in a Bitcoin block"""
        try:
            response = self.session.get(f"{self.node_url}/block/{block_hash}/txs")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Bitcoin block transactions for {block_hash}: {e}")
            return []
    
    def get_latest_blocks(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get latest Bitcoin blocks"""
        try:
            response = self.session.get(f"{self.node_url}/blocks")
            response.raise_for_status()
            blocks = response.json()
            return blocks[:count]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching latest Bitcoin blocks: {e}")
            return []


class EthereumAPIClient:
    """Ethereum blockchain node API client"""
    
    def __init__(self, node_url: Optional[str] = None, api_key: Optional[str] = None):
        self.node_url = node_url or os.getenv('ETHEREUM_NODE_URL', 'https://api.etherscan.io/api')
        self.api_key = api_key or os.getenv('ETHERSCAN_API_KEY')
        self.session = requests.Session()
        
    def get_transaction(self, txhash: str) -> Dict[str, Any]:
        """Get Ethereum transaction details"""
        try:
            params = {
                'module': 'proxy',
                'action': 'eth_getTransactionByHash',
                'txhash': txhash,
                'apikey': self.api_key
            }
            response = self.session.get(self.node_url, params=params)
            response.raise_for_status()
            return response.json().get('result', {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Ethereum transaction {txhash}: {e}")
            return {}
    
    def get_address_transactions(self, address: str, start_block: int = 0, 
                               end_block: int = 99999999, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions for an Ethereum address"""
        try:
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': address,
                'startblock': start_block,
                'endblock': end_block,
                'page': 1,
                'offset': limit,
                'sort': 'desc',
                'apikey': self.api_key
            }
            response = self.session.get(self.node_url, params=params)
            response.raise_for_status()
            return response.json().get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Ethereum address transactions for {address}: {e}")
            return []
    
    def get_token_transfers(self, address: str, contract_address: Optional[str] = None, 
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Get ERC-20 token transfers for an address"""
        try:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'address': address,
                'page': 1,
                'offset': limit,
                'sort': 'desc',
                'apikey': self.api_key
            }
            if contract_address:
                params['contractaddress'] = contract_address
                
            response = self.session.get(self.node_url, params=params)
            response.raise_for_status()
            return response.json().get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Ethereum token transfers for {address}: {e}")
            return []


class CoinbaseAPIClient:
    """Coinbase Pro API client for real-time market data and transactions"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None):
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        self.passphrase = passphrase or os.getenv('COINBASE_PASSPHRASE')
        self.base_url = 'https://api.exchange.coinbase.com'
        self.session = requests.Session()
    
    def _create_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Create authentication signature for Coinbase Pro API"""
        if not self.api_secret:
            return ''
        
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def _make_authenticated_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Coinbase Pro API"""
        try:
            timestamp = str(time.time())
            path = f"/{endpoint}"
            
            headers = {
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': self._create_signature(timestamp, method, path),
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
            
            url = f"{self.base_url}{path}"
            response = self.session.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinbase API error for {endpoint}: {e}")
            return {}
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Get user's Coinbase accounts"""
        result = self._make_authenticated_request('GET', 'accounts')
        return result if isinstance(result, list) else []
    
    def get_fills(self, product_id: str = 'BTC-USD', limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent fills (trades) for a product"""
        params = {'product_id': product_id, 'limit': limit}
        result = self._make_authenticated_request('GET', 'fills', params)
        return result if isinstance(result, list) else []
    
    def get_product_ticker(self, product_id: str = 'BTC-USD') -> Dict[str, Any]:
        """Get current ticker for a product"""
        try:
            response = self.session.get(f"{self.base_url}/products/{product_id}/ticker")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Coinbase ticker for {product_id}: {e}")
            return {}


class BinanceAPIClient:
    """Binance API client for market data and trading information"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.base_url = 'https://api.binance.com/api/v3'
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-MBX-APIKEY': self.api_key})
    
    def _create_signature(self, params: str) -> str:
        """Create signature for authenticated Binance API requests"""
        if not self.api_secret:
            return ''
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_account_trades(self, symbol: str = 'BTCUSDT', limit: int = 100) -> List[Dict[str, Any]]:
        """Get account trade history"""
        try:
            timestamp = int(time.time() * 1000)
            params = f"symbol={symbol}&limit={limit}&timestamp={timestamp}"
            signature = self._create_signature(params)
            
            url = f"{self.base_url}/myTrades?{params}&signature={signature}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Binance trades for {symbol}: {e}")
            return []
    
    def get_ticker_24hr(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """Get 24hr ticker price change statistics"""
        try:
            params = {'symbol': symbol}
            response = self.session.get(f"{self.base_url}/ticker/24hr", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Binance 24hr ticker for {symbol}: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str = 'BTCUSDT', limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol"""
        try:
            params = {'symbol': symbol, 'limit': limit}
            response = self.session.get(f"{self.base_url}/trades", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Binance recent trades for {symbol}: {e}")
            return []


class CrossChainAnalyzer:
    """Cross-chain transaction analysis and correlation"""
    
    def __init__(self):
        self.btc_client = BitcoinAPIClient()
        self.eth_client = EthereumAPIClient()
        self.coinbase_client = CoinbaseAPIClient()
        self.binance_client = BinanceAPIClient()
    
    def analyze_address_across_chains(self, btc_address: Optional[str] = None, eth_address: Optional[str] = None) -> Dict[str, Any]:
        """Analyze addresses across Bitcoin and Ethereum blockchains"""
        results = {
            'bitcoin_analysis': {},
            'ethereum_analysis': {},
            'cross_chain_patterns': {},
            'risk_indicators': []
        }
        
        # Bitcoin analysis
        if btc_address:
            btc_txs = self.btc_client.get_address_transactions(btc_address)
            results['bitcoin_analysis'] = {
                'address': btc_address,
                'transaction_count': len(btc_txs),
                'transactions': btc_txs[:10],  # Limit for performance
                'total_volume': sum(tx.get('value', 0) for tx in btc_txs) if btc_txs else 0
            }
        
        # Ethereum analysis
        if eth_address:
            eth_txs = self.eth_client.get_address_transactions(eth_address)
            token_txs = self.eth_client.get_token_transfers(eth_address)
            
            results['ethereum_analysis'] = {
                'address': eth_address,
                'transaction_count': len(eth_txs),
                'token_transfer_count': len(token_txs),
                'transactions': eth_txs[:10],  # Limit for performance
                'token_transfers': token_txs[:10],
                'total_volume_eth': sum(float(tx.get('value', 0)) / 1e18 for tx in eth_txs) if eth_txs else 0
            }
        
        # Cross-chain pattern analysis
        results['cross_chain_patterns'] = self._analyze_cross_chain_patterns(results)
        
        return results
    
    def _analyze_cross_chain_patterns(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across different blockchain networks"""
        patterns = {
            'timing_correlation': False,
            'amount_correlation': False,
            'suspicious_patterns': [],
            'risk_score': 0.0
        }
        
        btc_data = analysis_data.get('bitcoin_analysis', {})
        eth_data = analysis_data.get('ethereum_analysis', {})
        
        # Check for timing correlations
        if btc_data.get('transactions') and eth_data.get('transactions'):
            # Simple timing correlation check (can be enhanced)
            btc_times = [tx.get('status', {}).get('block_time', 0) for tx in btc_data['transactions']]
            eth_times = [int(tx.get('timeStamp', 0)) for tx in eth_data['transactions']]
            
            # Look for transactions within similar timeframes
            time_correlations = 0
            for btc_time in btc_times:
                for eth_time in eth_times:
                    if abs(btc_time - eth_time) < 3600:  # Within 1 hour
                        time_correlations += 1
            
            if time_correlations > 2:
                patterns['timing_correlation'] = True
                patterns['suspicious_patterns'].append('Multiple transactions across chains within similar timeframes')
                patterns['risk_score'] += 0.3
        
        # Check for amount patterns
        btc_volume = btc_data.get('total_volume', 0)
        eth_volume = eth_data.get('total_volume_eth', 0)
        
        if btc_volume > 0 and eth_volume > 0:
            # Check for similar large amounts (potential laundering)
            if btc_volume > 10 or eth_volume > 10:  # Large amounts
                patterns['amount_correlation'] = True
                patterns['suspicious_patterns'].append('Large volume transactions on multiple chains')
                patterns['risk_score'] += 0.4
        
        return patterns
    
    def get_exchange_correlation_data(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """Get correlation data from multiple exchanges"""
        results = {
            'coinbase_data': {},
            'binance_data': {},
            'price_analysis': {},
            'volume_analysis': {}
        }
        
        # Get Coinbase data
        coinbase_ticker = self.coinbase_client.get_product_ticker('BTC-USD')
        if coinbase_ticker:
            results['coinbase_data'] = {
                'price': float(coinbase_ticker.get('price', 0)),
                'volume': float(coinbase_ticker.get('volume', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        # Get Binance data
        binance_ticker = self.binance_client.get_ticker_24hr(symbol)
        if binance_ticker:
            results['binance_data'] = {
                'price': float(binance_ticker.get('lastPrice', 0)),
                'volume': float(binance_ticker.get('volume', 0)),
                'price_change': float(binance_ticker.get('priceChangePercent', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        # Analyze price and volume correlations
        results['price_analysis'] = self._analyze_exchange_prices(results)
        results['volume_analysis'] = self._analyze_exchange_volumes(results)
        
        return results
    
    def _analyze_exchange_prices(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price differences between exchanges"""
        coinbase_price = exchange_data.get('coinbase_data', {}).get('price', 0)
        binance_price = exchange_data.get('binance_data', {}).get('price', 0)
        
        analysis = {
            'price_difference': 0,
            'percentage_difference': 0,
            'arbitrage_opportunity': False
        }
        
        if coinbase_price > 0 and binance_price > 0:
            analysis['price_difference'] = abs(coinbase_price - binance_price)
            analysis['percentage_difference'] = (analysis['price_difference'] / min(coinbase_price, binance_price)) * 100
            analysis['arbitrage_opportunity'] = analysis['percentage_difference'] > 1.0  # >1% difference
        
        return analysis
    
    def _analyze_exchange_volumes(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns between exchanges"""
        coinbase_volume = exchange_data.get('coinbase_data', {}).get('volume', 0)
        binance_volume = exchange_data.get('binance_data', {}).get('volume', 0)
        
        analysis = {
            'total_volume': coinbase_volume + binance_volume,
            'volume_ratio': 0,
            'market_activity': 'Low'
        }
        
        if binance_volume > 0:
            analysis['volume_ratio'] = coinbase_volume / binance_volume
        
        if analysis['total_volume'] > 100000:
            analysis['market_activity'] = 'High'
        elif analysis['total_volume'] > 50000:
            analysis['market_activity'] = 'Medium'
        
        return analysis


def convert_blockchain_data_to_standard_format(data: List[Dict[str, Any]], source: str) -> pd.DataFrame:
    """Convert blockchain API data to standard QuantumGuard format"""
    
    if not data:
        return pd.DataFrame()
    
    if source.lower() == 'bitcoin':
        # Convert Bitcoin transaction format
        transactions = []
        for tx in data:
            if isinstance(tx, dict):
                transactions.append({
                    'from_address': tx.get('vin', [{}])[0].get('prevout', {}).get('scriptpubkey_address', 'Unknown') if tx.get('vin') else 'Unknown',
                    'to_address': tx.get('vout', [{}])[0].get('scriptpubkey_address', 'Unknown') if tx.get('vout') else 'Unknown',
                    'value': tx.get('vout', [{}])[0].get('value', 0) if tx.get('vout') else 0,
                    'timestamp': datetime.fromtimestamp(tx.get('status', {}).get('block_time', 0)).strftime('%Y-%m-%d %H:%M:%S') if tx.get('status', {}).get('block_time') else '',
                    'transaction_hash': tx.get('txid', ''),
                    'blockchain': 'Bitcoin',
                    'source': source
                })
        
        return pd.DataFrame(transactions)
    
    elif source.lower() == 'ethereum':
        # Convert Ethereum transaction format
        transactions = []
        for tx in data:
            if isinstance(tx, dict):
                transactions.append({
                    'from_address': tx.get('from', ''),
                    'to_address': tx.get('to', ''),
                    'value': float(tx.get('value', '0')) / 1e18,  # Convert from wei to ETH
                    'timestamp': datetime.fromtimestamp(int(tx.get('timeStamp', '0'))).strftime('%Y-%m-%d %H:%M:%S') if tx.get('timeStamp') else '',
                    'transaction_hash': tx.get('hash', ''),
                    'gas_price': int(tx.get('gasPrice', '0')),
                    'gas_used': int(tx.get('gasUsed', '0')),
                    'blockchain': 'Ethereum',
                    'source': source
                })
        
        return pd.DataFrame(transactions)
    
    else:
        # Generic format for other sources
        return pd.DataFrame(data)


# Initialize global clients for easy access
blockchain_api_clients = {
    'bitcoin': BitcoinAPIClient(),
    'ethereum': EthereumAPIClient(),
    'coinbase': CoinbaseAPIClient(),
    'binance': BinanceAPIClient(),
    'cross_chain': CrossChainAnalyzer()
}