"""
Test script to demonstrate enhanced bank transaction risk detection
"""

import pandas as pd
from bank_transaction_risk_analyzer import bank_risk_analyzer

def test_risk_detection():
    """Test the enhanced risk detection on sample data"""
    
    print("🔍 Testing Enhanced Bank Transaction Risk Detection\n")
    print("=" * 80)
    
    # Load sample datasets
    try:
        test_df = pd.read_csv('sample_test_100_transactions.csv')
        print(f"\n✅ Loaded test dataset: {len(test_df)} transactions")
    except:
        print("\n⚠️ Test dataset not found. Run sample_bank_transactions.py first")
        return
    
    # Generate comprehensive risk report
    print("\n📊 Analyzing transactions...\n")
    report = bank_risk_analyzer.generate_risk_report(test_df)
    
    # Display summary
    print("=" * 80)
    print("RISK ASSESSMENT SUMMARY")
    print("=" * 80)
    
    summary = report['summary']
    print(f"\n📈 Total Transactions: {summary['total_transactions']}")
    print(f"🚨 Flagged for Review: {summary['flagged_for_review']} ({summary['risk_percentage']:.1f}%)")
    print(f"📋 AUSTRAC Reporting Required: {summary['austrac_reporting_required']}")
    
    # Risk distribution
    print("\n📊 Risk Distribution:")
    for level, count in sorted(report['risk_distribution'].items()):
        percentage = (count / summary['total_transactions'] * 100)
        print(f"   {level:12s}: {count:3d} ({percentage:5.1f}%)")
    
    # Top risk flags
    print("\n🚩 Top Risk Flags Detected:")
    for flag, count in list(report['top_risk_flags'].items())[:10]:
        print(f"   {flag:35s}: {count:3d}")
    
    # Repetitive patterns
    if report['repetitive_patterns']:
        print(f"\n🔄 Repetitive Patterns Found: {len(report['repetitive_patterns'])}")
        for pattern in report['repetitive_patterns'][:3]:
            print(f"   - {pattern['occurrences']} transactions of ${pattern['amount']:.2f}")
            print(f"     Risk Score: {pattern['risk_score']}, Detail: {pattern['detail']}")
    
    # Velocity anomalies
    if report['velocity_anomalies']:
        print(f"\n⚡ Velocity Anomalies Found: {len(report['velocity_anomalies'])}")
        for anomaly in report['velocity_anomalies'][:3]:
            print(f"   - {anomaly['anomaly_type']}: {anomaly['detail']}")
    
    # High-risk transactions
    print(f"\n🚨 High-Risk Transactions (Sample):")
    high_risk = report['high_risk_transactions'][:5]
    
    for i, tx in enumerate(high_risk, 1):
        print(f"\n   Transaction #{i}:")
        print(f"   ID: {tx['transaction_id']}")
        print(f"   Risk Score: {tx['risk_score']:.1f}/100")
        print(f"   Risk Level: {tx['risk_level']}")
        print(f"   Flags: {', '.join(tx['risk_flags'])}")
        if tx['risk_details']:
            print(f"   Details:")
            for detail in tx['risk_details']:
                print(f"      • {detail}")
    
    # Export analyzed data
    analyzed_df = report['analyzed_data']
    analyzed_df.to_csv('analyzed_transactions.csv', index=False)
    print(f"\n💾 Full analysis saved to: analyzed_transactions.csv")
    
    print("\n" + "=" * 80)
    print("✅ Risk Detection Test Complete!")
    print("=" * 80)


def test_specific_patterns():
    """Test specific fraud patterns"""
    
    print("\n\n🧪 Testing Specific Fraud Patterns\n")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            'name': 'Late Night Transaction',
            'transaction': {
                'transaction_id': 'TEST_001',
                'amount': 500.0,
                'merchant': 'Star Casino',
                'timestamp': '2025-10-15T02:30:00',
                'country': 'Australia'
            }
        },
        {
            'name': 'Suspicious Merchant Name',
            'transaction': {
                'transaction_id': 'TEST_002',
                'amount': 2500.0,
                'merchant': 'UBER BARANGAROO INCORPORATED',
                'timestamp': '2025-10-15T14:30:00',
                'country': 'Australia'
            }
        },
        {
            'name': 'Tax Haven Transaction',
            'transaction': {
                'transaction_id': 'TEST_003',
                'amount': 15000.0,
                'merchant': 'Offshore Holdings',
                'timestamp': '2025-10-15T10:00:00',
                'country': 'Cayman Islands'
            }
        },
        {
            'name': 'Structuring Attempt',
            'transaction': {
                'transaction_id': 'TEST_004',
                'amount': 9500.0,
                'merchant': 'Transfer Service',
                'timestamp': '2025-10-15T15:00:00',
                'country': 'Australia'
            }
        },
        {
            'name': 'Combined High Risk',
            'transaction': {
                'transaction_id': 'TEST_005',
                'amount': 9800.0,
                'merchant': 'CRYPTO OFFSHORE INCORPORATED',
                'timestamp': '2025-10-15T01:45:00',
                'country': 'Panama'
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Test: {test_case['name']}")
        print(f"{'─' * 80}")
        
        result = bank_risk_analyzer.analyze_transaction(test_case['transaction'])
        
        print(f"\n📊 Analysis Result:")
        print(f"   Risk Score: {result['risk_score']:.1f}/100")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Flagged for Review: {'YES' if result['flagged_for_review'] else 'NO'}")
        print(f"   AUSTRAC Reporting: {'REQUIRED' if result['austrac_reporting_required'] else 'Not Required'}")
        
        if result['risk_flags']:
            print(f"\n   🚩 Risk Flags:")
            for flag in result['risk_flags']:
                print(f"      • {flag}")
        
        if result['risk_details']:
            print(f"\n   📋 Details:")
            for detail in result['risk_details']:
                print(f"      • {detail}")
    
    print(f"\n{'=' * 80}")
    print("✅ Pattern Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_risk_detection()
    test_specific_patterns()
    
    print("\n\n💡 Next Steps:")
    print("   1. Upload the generated CSV files to the main app")
    print("   2. The enhanced detection will automatically activate")
    print("   3. Review analyzed_transactions.csv for detailed results")
    print("   4. Use these patterns to train your team on fraud detection\n")
