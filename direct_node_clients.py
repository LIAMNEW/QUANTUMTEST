import json
import requests
import os
import base64
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinNodeClient:
    """Direct Bitcoin Core (bitcoind) JSON-RPC client for blockchain node connections"""
    
    def __init__(self, node_url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        self.node_url = node_url or os.getenv('BITCOIN_NODE_URL', 'http://localhost:8332')
        self.username = username or os.getenv('BITCOIN_RPC_USER', 'bitcoin')
        self.password = password or os.getenv('BITCOIN_RPC_PASSWORD', '')
        self.session = requests.Session()
        
        # Set up basic authentication for bitcoind
        if self.username and self.password:
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            self.session.headers.update({
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/json'
            })
    
    def _make_rpc_call(self, method: str, params: List = None) -> Dict[str, Any]:
        """Make a JSON-RPC call to Bitcoin Core node"""
        if params is None:
            params = []
            
        payload = {
            "jsonrpc": "2.0",
            "id": "quantumguard",
            "method": method,
            "params": params
        }
        
        try:
            response = self.session.post(self.node_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get('error'):
                logger.error(f"Bitcoin RPC error: {result['error']}")
                return {}
            
            return result.get('result', {})
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Bitcoin node connection error: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Bitcoin RPC JSON decode error: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test connection to Bitcoin node"""
        try:
            result = self._make_rpc_call('getblockchaininfo')
            return bool(result.get('chain'))
        except Exception:
            return False
    
    def get_transaction(self, txid: str) -> Dict[str, Any]:
        """Get Bitcoin transaction by transaction ID"""
        return self._make_rpc_call('getrawtransaction', [txid, True])
    
    def get_address_transactions(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions for Bitcoin address (requires address index)"""
        # Note: This requires Bitcoin Core with address index enabled (-addressindex)
        try:
            # Get address history using listreceivedbyaddress for received transactions
            received = self._make_rpc_call('listreceivedbyaddress', [1, True, True, address])
            
            transactions = []
            for entry in received:
                if entry.get('address') == address:
                    for txid in entry.get('txids', []):
                        tx_data = self.get_transaction(txid)
                        if tx_data:
                            transactions.append(tx_data)
                            if len(transactions) >= limit:
                                break
                
            return transactions[:limit]
        
        except Exception as e:
            logger.error(f"Error getting address transactions: {e}")
            return []
    
    def get_block_transactions(self, block_hash: str) -> List[Dict[str, Any]]:
        """Get all transactions in a Bitcoin block"""
        try:
            block_data = self._make_rpc_call('getblock', [block_hash, 2])  # Verbosity 2 for full transaction data
            return block_data.get('tx', [])
        except Exception as e:
            logger.error(f"Error getting block transactions: {e}")
            return []
    
    def get_latest_blocks(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get latest Bitcoin blocks"""
        try:
            # Get current block height
            blockchain_info = self._make_rpc_call('getblockchaininfo')
            current_height = blockchain_info.get('blocks', 0)
            
            blocks = []
            for i in range(count):
                if current_height - i >= 0:
                    block_hash = self._make_rpc_call('getblockhash', [current_height - i])
                    if block_hash:
                        block_data = self._make_rpc_call('getblock', [block_hash, 1])
                        if block_data:
                            blocks.append(block_data)
            
            return blocks
        except Exception as e:
            logger.error(f"Error getting latest blocks: {e}")
            return []
    
    def get_mempool_transactions(self, limit: int = 20) -> List[str]:
        """Get transactions from mempool"""
        try:
            mempool_txids = self._make_rpc_call('getrawmempool')
            return mempool_txids[:limit] if isinstance(mempool_txids, list) else []
        except Exception as e:
            logger.error(f"Error getting mempool transactions: {e}")
            return []


class EthereumNodeClient:
    """Direct Ethereum node (geth/web3) JSON-RPC client for blockchain node connections"""
    
    def __init__(self, node_url: Optional[str] = None):
        self.node_url = node_url or os.getenv('ETHEREUM_NODE_URL', 'http://localhost:8545')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _make_rpc_call(self, method: str, params: List = None) -> Any:
        """Make a JSON-RPC call to Ethereum node"""
        if params is None:
            params = []
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = self.session.post(self.node_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get('error'):
                logger.error(f"Ethereum RPC error: {result['error']}")
                return None
            
            return result.get('result')
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ethereum node connection error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Ethereum RPC JSON decode error: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to Ethereum node"""
        try:
            result = self._make_rpc_call('web3_clientVersion')
            return result is not None
        except Exception:
            return False
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get Ethereum transaction by hash"""
        result = self._make_rpc_call('eth_getTransactionByHash', [tx_hash])
        return result if result else {}
    
    def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """Get Ethereum transaction receipt"""
        result = self._make_rpc_call('eth_getTransactionReceipt', [tx_hash])
        return result if result else {}
    
    def get_block_transactions(self, block_number: Union[int, str]) -> List[Dict[str, Any]]:
        """Get all transactions in an Ethereum block"""
        try:
            # Convert block number to hex if it's an integer
            if isinstance(block_number, int):
                block_number = hex(block_number)
            elif isinstance(block_number, str) and block_number.isdigit():
                block_number = hex(int(block_number))
            
            block_data = self._make_rpc_call('eth_getBlockByNumber', [block_number, True])
            if block_data and 'transactions' in block_data:
                return block_data['transactions']
            
            return []
        except Exception as e:
            logger.error(f"Error getting block transactions: {e}")
            return []
    
    def get_latest_block(self) -> Dict[str, Any]:
        """Get latest Ethereum block"""
        result = self._make_rpc_call('eth_getBlockByNumber', ['latest', True])
        return result if result else {}
    
    def get_balance(self, address: str, block: str = 'latest') -> str:
        """Get Ethereum address balance"""
        result = self._make_rpc_call('eth_getBalance', [address, block])
        return result if result else '0x0'
    
    def get_transaction_count(self, address: str, block: str = 'latest') -> int:
        """Get transaction count for address (nonce)"""
        result = self._make_rpc_call('eth_getTransactionCount', [address, block])
        return int(result, 16) if result else 0
    
    def get_logs(self, from_block: str = 'latest', to_block: str = 'latest', 
                 address: Optional[str] = None, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get Ethereum logs"""
        filter_params = {
            'fromBlock': from_block,
            'toBlock': to_block
        }
        
        if address:
            filter_params['address'] = address
        if topics:
            filter_params['topics'] = topics
        
        result = self._make_rpc_call('eth_getLogs', [filter_params])
        return result if isinstance(result, list) else []
    
    def get_pending_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get pending transactions from mempool"""
        try:
            pending_block = self._make_rpc_call('eth_getBlockByNumber', ['pending', True])
            if pending_block and 'transactions' in pending_block:
                return pending_block['transactions'][:limit]
            return []
        except Exception as e:
            logger.error(f"Error getting pending transactions: {e}")
            return []


class EnhancedCoinbaseClient:
    """Enhanced Coinbase API client with fixed authentication"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None):
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        self.passphrase = passphrase or os.getenv('COINBASE_PASSPHRASE')
        self.base_url = 'https://api.exchange.coinbase.com'
        self.session = requests.Session()
    
    def _create_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Create properly formatted authentication signature"""
        if not self.api_secret:
            return ''
        
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def _make_authenticated_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                                   json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request with proper signature including query parameters"""
        try:
            import time
            timestamp = str(time.time())
            
            # Build the full path including query parameters
            path = f"/{endpoint.lstrip('/')}"
            if params:
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                path += f"?{query_string}"
            
            # Prepare body for signature
            body = json.dumps(json_data) if json_data else ''
            
            # Create signature with the full path including query parameters
            signature = self._create_signature(timestamp, method, path, body)
            
            headers = {
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
            
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.request(
                method, 
                url, 
                headers=headers, 
                params=params,
                json=json_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinbase API error for {endpoint}: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test Coinbase API connection with authenticated endpoint"""
        try:
            accounts = self._make_authenticated_request('GET', 'accounts')
            return isinstance(accounts, list)
        except Exception:
            # Fallback to public endpoint
            try:
                response = self.session.get(f"{self.base_url}/products/BTC-USD/ticker")
                return response.status_code == 200
            except Exception:
                return False
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Get user's Coinbase accounts"""
        result = self._make_authenticated_request('GET', 'accounts')
        return result if isinstance(result, list) else []
    
    def get_fills(self, product_id: str = 'BTC-USD', limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent fills (trades) for a product"""
        params = {'product_id': product_id, 'limit': limit}
        result = self._make_authenticated_request('GET', 'fills', params=params)
        return result if isinstance(result, list) else []
    
    def get_product_ticker(self, product_id: str = 'BTC-USD') -> Dict[str, Any]:
        """Get current ticker for a product (public endpoint)"""
        try:
            response = self.session.get(f"{self.base_url}/products/{product_id}/ticker")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Coinbase ticker for {product_id}: {e}")
            return {}


class NodeConnectionManager:
    """Manages connections to different blockchain nodes and APIs"""
    
    def __init__(self):
        self.bitcoin_node = BitcoinNodeClient()
        self.ethereum_node = EthereumNodeClient()
        self.coinbase_client = EnhancedCoinbaseClient()
        
        # Import REST clients as fallback
        try:
            from blockchain_api_integrations import BitcoinAPIClient, EthereumAPIClient
            self.bitcoin_rest = BitcoinAPIClient()
            self.ethereum_rest = EthereumAPIClient()
        except ImportError:
            self.bitcoin_rest = None
            self.ethereum_rest = None
    
    def get_bitcoin_client(self) -> Union[BitcoinNodeClient, Any]:
        """Get Bitcoin client, preferring direct node connection"""
        if self.bitcoin_node.test_connection():
            logger.info("Using direct Bitcoin node connection")
            return self.bitcoin_node
        elif self.bitcoin_rest:
            logger.info("Falling back to Bitcoin REST API")
            return self.bitcoin_rest
        else:
            logger.warning("No Bitcoin client available")
            return self.bitcoin_node  # Return node client anyway
    
    def get_ethereum_client(self) -> Union[EthereumNodeClient, Any]:
        """Get Ethereum client, preferring direct node connection"""
        if self.ethereum_node.test_connection():
            logger.info("Using direct Ethereum node connection")
            return self.ethereum_node
        elif self.ethereum_rest:
            logger.info("Falling back to Ethereum REST API")
            return self.ethereum_rest
        else:
            logger.warning("No Ethereum client available")
            return self.ethereum_node  # Return node client anyway
    
    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test all blockchain connections"""
        results = {}
        
        # Test Bitcoin connections
        btc_node_status = self.bitcoin_node.test_connection()
        btc_rest_status = self.bitcoin_rest is not None
        
        results['Bitcoin'] = {
            'direct_node': btc_node_status,
            'rest_api': btc_rest_status,
            'preferred': 'Direct Node' if btc_node_status else 'REST API',
            'status': 'success' if (btc_node_status or btc_rest_status) else 'failed'
        }
        
        # Test Ethereum connections
        eth_node_status = self.ethereum_node.test_connection()
        eth_rest_status = self.ethereum_rest is not None
        
        results['Ethereum'] = {
            'direct_node': eth_node_status,
            'rest_api': eth_rest_status,
            'preferred': 'Direct Node' if eth_node_status else 'REST API',
            'status': 'success' if (eth_node_status or eth_rest_status) else 'failed'
        }
        
        # Test Coinbase
        coinbase_status = self.coinbase_client.test_connection()
        results['Coinbase'] = {
            'status': 'success' if coinbase_status else 'warning',
            'message': 'Connected' if coinbase_status else 'Public access only'
        }
        
        return results
    
    def get_connection_info(self) -> Dict[str, str]:
        """Get information about current connections"""
        return {
            'bitcoin_node_url': self.bitcoin_node.node_url,
            'ethereum_node_url': self.ethereum_node.node_url,
            'coinbase_url': self.coinbase_client.base_url
        }


# Initialize global node manager for easy access
node_manager = NodeConnectionManager()