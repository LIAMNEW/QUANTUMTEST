"""
Sample Bank Transaction Generator for Testing QuantumGuard AI
Creates realistic test datasets with obvious fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_dataset_100():
    """Generate 100-transaction test dataset with obvious fraud cases"""
    
    transactions = []
    base_date = datetime.now() - timedelta(days=30)
    
    # 20 Normal transactions
    for i in range(20):
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(hours=i*12 + random.randint(9, 18))).isoformat(),
            'amount': random.uniform(15, 200),
            'merchant': random.choice(['Woolworths', 'Coles', 'Shell', 'BP', 'Target', 'Myer']),
            'country': 'Australia',
            'description': 'Normal purchase'
        })
    
    # 15 Late-night transactions (HIGH RISK)
    for i in range(20, 35):
        hour = random.randint(0, 4)  # 12am-5am
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i) + timedelta(hours=hour)).isoformat(),
            'amount': random.uniform(50, 500),
            'merchant': random.choice(['7-Eleven', 'Night Owl', 'Star Casino', 'Crown Casino']),
            'country': 'Australia',
            'description': 'Late night transaction'
        })
    
    # 15 Suspicious merchant names (HIGH RISK)
    suspicious_merchants = [
        'UBER BARANGAROO INCORPORATED',
        'XYZ1234 HOLDINGS PTY LTD',
        'CRYPTO TRANSFER SERVICE INTERNATIONAL',
        'OFFSHORE GLOBAL VENTURES LLC',
        'SHELL NOMINEES PTY LTD',
        'A B SMITH INCORPORATED',
        'BITCOIN EXCHANGE 9999',
        'RANDOM4567 CORP',
        'CASINO INTERNATIONAL HOLDINGS'
    ]
    
    for i in range(35, 50):
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i)).isoformat(),
            'amount': random.uniform(100, 2000),
            'merchant': random.choice(suspicious_merchants),
            'country': 'Australia',
            'description': 'Suspicious merchant'
        })
    
    # 20 Foreign country transactions (MEDIUM-HIGH RISK)
    high_risk_countries = ['Cayman Islands', 'Panama', 'Russia', 'North Korea', 'Iran']
    medium_risk_countries = ['Nigeria', 'Philippines', 'Turkey', 'UAE', 'Pakistan']
    
    for i in range(50, 70):
        country = random.choice(high_risk_countries + medium_risk_countries)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i)).isoformat(),
            'amount': random.uniform(500, 5000),
            'merchant': f'International Merchant {i}',
            'country': country,
            'description': f'Transaction from {country}'
        })
    
    # 15 Structuring attempts - amounts just below $10,000 (CRITICAL RISK)
    for i in range(70, 85):
        # Amounts between $8,000 and $9,900 (below AUSTRAC $10,000 threshold)
        amount = random.uniform(8000, 9900)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i)).isoformat(),
            'amount': amount,
            'merchant': 'Generic Transfer Service',
            'country': 'Australia',
            'description': f'Potential structuring - ${amount:.2f}'
        })
    
    # 15 Repetitive small amounts (STRUCTURING PATTERN)
    base_amount = 2500.0
    for i in range(85, 100):
        # All similar amounts around $2,500
        amount = base_amount + random.uniform(-50, 50)
        hour = random.randint(1, 3)  # Late night
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(hours=i*2 + hour)).isoformat(),
            'amount': amount,
            'merchant': 'Money Transfer Service',
            'country': 'Australia',
            'description': f'Repetitive pattern #{i-84}'
        })
    
    df = pd.DataFrame(transactions)
    return df


def generate_realistic_dataset_1000():
    """Generate 1000-transaction realistic dataset"""
    
    transactions = []
    base_date = datetime.now() - timedelta(days=90)
    
    # 700 Normal transactions (70%)
    normal_merchants = [
        'Woolworths', 'Coles', 'IGA', 'Aldi',
        'Shell', 'BP', 'Caltex', '7-Eleven',
        'McDonald\'s', 'KFC', 'Subway', 'Hungry Jack\'s',
        'Dan Murphy\'s', 'BWS', 'Liquorland',
        'JB Hi-Fi', 'Harvey Norman', 'The Good Guys',
        'Bunnings', 'Officeworks', 'Kmart', 'Target',
        'Netflix', 'Spotify', 'Apple',
        'Commonwealth Bank', 'Westpac', 'NAB'
    ]
    
    for i in range(700):
        hour = random.randint(7, 21)  # Normal business hours
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//8, hours=hour)).isoformat(),
            'amount': abs(np.random.normal(85, 120)),  # Average $85, some variation
            'merchant': random.choice(normal_merchants),
            'country': 'Australia',
            'description': 'Normal transaction'
        })
    
    # 50 Late-night transactions (5%)
    for i in range(700, 750):
        hour = random.choice([0, 1, 2, 3, 4, 23])  # Late night/early morning
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//3, hours=hour)).isoformat(),
            'amount': random.uniform(20, 800),
            'merchant': random.choice(['Star City Casino', 'Crown Perth', '24/7 Convenience', 'Night Owl']),
            'country': 'Australia',
            'description': 'Late night transaction'
        })
    
    # 80 International transactions (8%)
    countries = ['USA', 'UK', 'Singapore', 'New Zealand', 'Japan', 'China', 'India',
                'Cayman Islands', 'Panama', 'UAE', 'Russia', 'Nigeria', 'Philippines']
    
    for i in range(750, 830):
        country = random.choice(countries)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//4)).isoformat(),
            'amount': random.uniform(100, 3000),
            'merchant': f'{country} Merchant {i}',
            'country': country,
            'description': f'International - {country}'
        })
    
    # 50 Suspicious merchants (5%)
    suspicious = [
        'CRYPTO EXCHANGE INTERNATIONAL LLC',
        'XYZ HOLDINGS 4456 PTY LTD',
        'UBER PARRAMATTA INCORPORATED',
        'OFFSHORE GLOBAL SERVICES',
        'RANDOM8899 TRANSFER SERVICE',
        'SHELL COMPANY NOMINEES',
        'BITCOIN VENTURES INCORPORATED',
        'ABC DEF GHI INTERNATIONAL',
        'GAMING HOLDINGS INTERNATIONAL'
    ]
    
    for i in range(830, 880):
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//5)).isoformat(),
            'amount': random.uniform(500, 5000),
            'merchant': random.choice(suspicious),
            'country': random.choice(['Australia', 'Cayman Islands', 'Panama']),
            'description': 'Suspicious merchant pattern'
        })
    
    # 40 Structuring attempts (4%)
    for i in range(880, 920):
        amount = random.uniform(8000, 9800)  # Just below $10k threshold
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//6)).isoformat(),
            'amount': amount,
            'merchant': 'Wire Transfer Service',
            'country': 'Australia',
            'description': 'Potential structuring'
        })
    
    # 30 Repetitive pattern (3%)
    for i in range(920, 950):
        amount = 3500 + random.uniform(-100, 100)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(hours=i*4)).isoformat(),
            'amount': amount,
            'merchant': 'Payment Processor',
            'country': 'Australia',
            'description': 'Repetitive pattern'
        })
    
    # 30 Large transactions (3%)
    for i in range(950, 980):
        amount = random.uniform(50000, 150000)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//10)).isoformat(),
            'amount': amount,
            'merchant': random.choice(['Property Purchase', 'Vehicle Dealer', 'Investment Fund']),
            'country': 'Australia',
            'description': 'Large transaction'
        })
    
    # 20 Combined high-risk (2%)
    for i in range(980, 1000):
        hour = random.randint(0, 3)
        amount = random.uniform(8500, 9500)
        transactions.append({
            'transaction_id': f'TX_{i+1:06d}',
            'date': (base_date + timedelta(days=i//12, hours=hour)).isoformat(),
            'amount': amount,
            'merchant': random.choice(['CRYPTO OFFSHORE INCORPORATED', 'SHELL HOLDINGS 9999']),
            'country': random.choice(['Cayman Islands', 'Panama', 'UAE']),
            'description': 'Multiple risk factors'
        })
    
    df = pd.DataFrame(transactions)
    return df


def save_sample_datasets():
    """Generate and save sample datasets"""
    
    # Generate test dataset (100 transactions - obvious cases)
    test_df = generate_test_dataset_100()
    test_df.to_csv('sample_test_100_transactions.csv', index=False)
    print("✅ Created: sample_test_100_transactions.csv (100 obvious test cases)")
    
    # Generate realistic dataset (1000 transactions)
    realistic_df = generate_realistic_dataset_1000()
    realistic_df.to_csv('sample_realistic_1000_transactions.csv', index=False)
    print("✅ Created: sample_realistic_1000_transactions.csv (1000 realistic transactions)")
    
    # Print summaries
    print("\n📊 Test Dataset (100 transactions):")
    print(f"  - 20 normal transactions")
    print(f"  - 15 late-night (12am-5am)")
    print(f"  - 15 suspicious merchants")
    print(f"  - 20 foreign countries")
    print(f"  - 15 structuring attempts")
    print(f"  - 15 repetitive patterns")
    
    print("\n📊 Realistic Dataset (1000 transactions):")
    print(f"  - 700 normal (70%)")
    print(f"  - 50 late-night (5%)")
    print(f"  - 80 international (8%)")
    print(f"  - 50 suspicious merchants (5%)")
    print(f"  - 40 structuring (4%)")
    print(f"  - 30 repetitive (3%)")
    print(f"  - 30 large amounts (3%)")
    print(f"  - 20 combined high-risk (2%)")
    
    return test_df, realistic_df


if __name__ == "__main__":
    save_sample_datasets()
