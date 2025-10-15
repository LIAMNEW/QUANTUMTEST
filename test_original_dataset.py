"""
Test the enhanced risk detection with the original AUSTRAC v2 dataset
"""

import pandas as pd
from bank_transaction_risk_analyzer import bank_risk_analyzer

# Load the original dataset with separate Date/Time columns
df = pd.read_csv('attached_assets/transaction_dataset_1000_AUSTRAC_v2_1760571217860.csv')

print("🔍 Testing Enhanced Risk Detection on Original Dataset\n")
print("=" * 80)
print(f"Dataset Info:")
print(f"  Columns: {', '.join(df.columns[:5])}...")
print(f"  Total Transactions: {len(df)}")
print(f"  Sample Date: {df['Date'].iloc[0]}, Time: {df['Time'].iloc[0]}")
print("=" * 80)

# Generate comprehensive risk report
print("\n📊 Analyzing transactions...\n")
report = bank_risk_analyzer.generate_risk_report(df)

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

# Check for late-night transactions specifically
analyzed_df = report['analyzed_data']
if 'risk_flags' in analyzed_df.columns:
    late_night_count = sum(1 for flags in analyzed_df['risk_flags'] if 'LATE_NIGHT_TRANSACTION' in flags)
    print(f"\n⏰ Late-Night Transactions (12am-5am): {late_night_count}")

# Show some high-risk examples
print(f"\n🚨 High-Risk Transaction Examples:")
high_risk = report['high_risk_transactions'][:3]

for i, tx in enumerate(high_risk, 1):
    print(f"\n   Transaction #{i}:")
    print(f"   ID: {tx.get('TransactionID', tx.get('transaction_id', 'N/A'))}")
    print(f"   Risk Score: {tx['risk_score']:.1f}/100")
    print(f"   Risk Level: {tx['risk_level']}")
    print(f"   Flags: {', '.join(tx['risk_flags'])}")
    if tx['risk_details']:
        for detail in tx['risk_details'][:2]:
            print(f"      • {detail}")

print("\n" + "=" * 80)
print("✅ Original Dataset Analysis Complete!")
print("=" * 80)

# Export results
analyzed_df.to_csv('original_dataset_analyzed.csv', index=False)
print(f"\n💾 Full analysis saved to: original_dataset_analyzed.csv")
