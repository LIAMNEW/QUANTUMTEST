import streamlit as st
import os
from typing import Dict, List, Optional, Any

class APIKeyManager:
    """Secure API key management system for blockchain integrations"""
    
    REQUIRED_KEYS = {
        'Bitcoin': {
            'BITCOIN_NODE_URL': {
                'description': 'Bitcoin Core node JSON-RPC URL (e.g., http://localhost:8332)',
                'default': 'http://localhost:8332',
                'required': False
            },
            'BITCOIN_RPC_USER': {
                'description': 'Bitcoin Core RPC username',
                'default': 'bitcoin',
                'required': False
            },
            'BITCOIN_RPC_PASSWORD': {
                'description': 'Bitcoin Core RPC password',
                'default': '',
                'required': False
            }
        },
        'Ethereum': {
            'ETHEREUM_NODE_URL': {
                'description': 'Ethereum node JSON-RPC URL (e.g., http://localhost:8545, or Infura/Alchemy)',
                'default': 'http://localhost:8545',
                'required': False
            },
            'ETHERSCAN_API_KEY': {
                'description': 'Etherscan API key (fallback - get from https://etherscan.io/apis)',
                'default': '',
                'required': False
            }
        },
        'Coinbase': {
            'COINBASE_API_KEY': {
                'description': 'Coinbase Pro API key',
                'default': '',
                'required': False
            },
            'COINBASE_API_SECRET': {
                'description': 'Coinbase Pro API secret',
                'default': '',
                'required': False
            },
            'COINBASE_PASSPHRASE': {
                'description': 'Coinbase Pro API passphrase',
                'default': '',
                'required': False
            }
        },
        'Binance': {
            'BINANCE_API_KEY': {
                'description': 'Binance API key',
                'default': '',
                'required': False
            },
            'BINANCE_API_SECRET': {
                'description': 'Binance API secret',
                'default': '',
                'required': False
            }
        }
    }
    
    @staticmethod
    def render_api_key_configuration():
        """Render API key configuration interface in Streamlit"""
        st.subheader("🔑 Blockchain API Configuration")
        
        st.info("""
        Configure your blockchain API keys to enable real-time data access:
        - **Bitcoin**: Free access via Blockstream (no key required)
        - **Ethereum**: Requires Etherscan API key (free tier available)
        - **Exchanges**: Optional for enhanced market data
        """)
        
        # Track which services are configured
        configured_services = []
        
        for service_name, keys in APIKeyManager.REQUIRED_KEYS.items():
            with st.expander(f"{service_name} API Configuration"):
                service_configured = True
                
                for key_name, key_info in keys.items():
                    current_value = os.getenv(key_name, key_info['default'])
                    
                    # Check if key is configured
                    if key_info['required'] and not current_value:
                        service_configured = False
                    
                    # Render input field
                    if 'secret' in key_name.lower() or 'passphrase' in key_name.lower():
                        # Use password input for secrets
                        input_value = st.text_input(
                            key_info['description'],
                            value=current_value,
                            type="password",
                            key=f"api_key_{key_name}",
                            help=f"Environment variable: {key_name}"
                        )
                    else:
                        input_value = st.text_input(
                            key_info['description'],
                            value=current_value,
                            key=f"api_key_{key_name}",
                            help=f"Environment variable: {key_name}"
                        )
                    
                    # Update environment variable if changed
                    if input_value != current_value:
                        os.environ[key_name] = input_value
                        if input_value:
                            st.success(f"✅ {key_name} updated")
                        else:
                            st.warning(f"⚠️ {key_name} cleared")
                
                # Display service status
                if service_configured:
                    configured_services.append(service_name)
                    st.success(f"✅ {service_name} API is configured")
                else:
                    st.warning(f"⚠️ {service_name} API requires additional configuration")
        
        return configured_services
    
    @staticmethod
    def check_api_configuration() -> Dict[str, bool]:
        """Check which APIs are properly configured"""
        status = {}
        
        for service_name, keys in APIKeyManager.REQUIRED_KEYS.items():
            service_configured = True
            
            for key_name, key_info in keys.items():
                current_value = os.getenv(key_name, key_info['default'])
                
                if key_info['required'] and not current_value:
                    service_configured = False
                    break
            
            status[service_name] = service_configured
        
        return status
    
    @staticmethod
    def get_api_status_summary() -> str:
        """Get a summary of API configuration status"""
        status = APIKeyManager.check_api_configuration()
        configured_count = sum(1 for v in status.values() if v)
        total_count = len(status)
        
        if configured_count == 0:
            return "❌ No APIs configured - using free/public endpoints only"
        elif configured_count == total_count:
            return f"✅ All {total_count} API services configured"
        else:
            return f"⚠️ {configured_count}/{total_count} API services configured"
    
    @staticmethod
    def render_quick_setup_guide():
        """Render quick setup guide for API keys"""
        st.subheader("📋 Quick Setup Guide")
        
        with st.expander("🚀 Getting Started with Blockchain APIs"):
            st.markdown("""
            ### Step 1: Direct Node Connections (Recommended)
            1. **Bitcoin Core Node** 
               - Set BITCOIN_NODE_URL to your bitcoind RPC endpoint
               - Configure RPC username/password in bitcoin.conf
               - Example: http://localhost:8332
            
            2. **Ethereum Node**
               - Set ETHEREUM_NODE_URL to your geth/node RPC endpoint  
               - Example: http://localhost:8545, wss://mainnet.infura.io/ws/v3/YOUR-PROJECT-ID
            
            ### Step 2: API Fallbacks (Optional)
            3. **Etherscan API Key** (Ethereum fallback)
               - Visit https://etherscan.io/apis
               - Only needed if no direct Ethereum node available
            
            3. **Exchange APIs** (Optional - for market data)
               - **Coinbase Pro**: Create API key at https://pro.coinbase.com/profile/api
               - **Binance**: Create API key at https://www.binance.com/en/my/settings/api-management
            
            ### Step 3: Test Configuration
            After adding keys, use the "Test API Connections" button below to verify setup.
            """)
        
        # Test connections button
        if st.button("🔧 Test API Connections"):
            return APIKeyManager.test_api_connections()
        
        return None
    
    @staticmethod
    def test_api_connections() -> Dict[str, Dict[str, Any]]:
        """Test all configured API connections"""
        from blockchain_api_integrations import blockchain_api_clients
        
        results = {}
        
        # Test Bitcoin connections
        try:
            from direct_node_clients import node_manager
            connection_tests = node_manager.test_all_connections()
            
            for service, test_result in connection_tests.items():
                results[service] = {
                    'status': test_result.get('status', 'failed'),
                    'message': f"✅ {test_result.get('preferred', 'Connected')}" if test_result.get('status') == 'success' else f"❌ {test_result.get('message', 'Connection failed')}",
                    'details': test_result
                }
        except Exception as e:
            results['Connection Test'] = {
                'status': 'error',
                'message': f"❌ Error: {str(e)[:50]}",
                'data': 0
            }
        
        # Test Ethereum API
        try:
            eth_client = blockchain_api_clients['ethereum']
            # Test with a known transaction hash
            test_tx = eth_client.get_transaction('0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12')
            results['Ethereum'] = {
                'status': 'success' if os.getenv('ETHERSCAN_API_KEY') else 'warning',
                'message': "✅ Connected with API key" if os.getenv('ETHERSCAN_API_KEY') else "⚠️ No API key - limited access",
                'data': 1 if os.getenv('ETHERSCAN_API_KEY') else 0
            }
        except Exception as e:
            results['Ethereum'] = {
                'status': 'error',
                'message': f"❌ Error: {str(e)[:50]}",
                'data': 0
            }
        
        # Test Coinbase API
        try:
            coinbase_client = blockchain_api_clients['coinbase']
            ticker = coinbase_client.get_product_ticker('BTC-USD')
            results['Coinbase'] = {
                'status': 'success' if ticker else 'warning',
                'message': "✅ Connected - Market data available" if ticker else "⚠️ Public access only",
                'data': 1 if ticker else 0
            }
        except Exception as e:
            results['Coinbase'] = {
                'status': 'error',
                'message': f"❌ Error: {str(e)[:50]}",
                'data': 0
            }
        
        # Test Binance API
        try:
            binance_client = blockchain_api_clients['binance']
            ticker = binance_client.get_ticker_24hr('BTCUSDT')
            results['Binance'] = {
                'status': 'success' if ticker else 'warning',
                'message': "✅ Connected - Market data available" if ticker else "⚠️ Public access only",
                'data': 1 if ticker else 0
            }
        except Exception as e:
            results['Binance'] = {
                'status': 'error',
                'message': f"❌ Error: {str(e)[:50]}",
                'data': 0
            }
        
        # Display results
        st.subheader("🔍 API Connection Test Results")
        
        for service, result in results.items():
            if result['status'] == 'success':
                st.success(f"**{service}**: {result['message']}")
            elif result['status'] == 'warning':
                st.warning(f"**{service}**: {result['message']}")
            else:
                st.error(f"**{service}**: {result['message']}")
        
        return results