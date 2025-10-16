"""
Test the account statement with 100 transactions
"""

import pandas as pd
from bank_transaction_risk_analyzer import bank_risk_analyzer

# Load the account statement
df = pd.read_csv('attached_assets/account_statement_100_1760573773832.csv')

print("🔍 Testing Account Statement (100 transactions)\n")
print("=" * 80)
print(f"Columns: {', '.join(df.columns)}")
print(f"Total Transactions: {len(df)}")
print(f"\nSample transactions:")
print(df[['TransactionID', 'Date', 'Time', 'Amount', 'Merchant', 'Country']].head(3))
print("=" * 80)

# Generate risk report
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
for flag, count in list(report['top_risk_flags'].items())[:15]:
    print(f"   {flag:35s}: {count:3d}")

# Show high-risk examples
print(f"\n🚨 High-Risk Transactions (Top 10):")
high_risk = report['high_risk_transactions'][:10]

for i, tx in enumerate(high_risk, 1):
    print(f"\n   #{i}: {tx.get('TransactionID', 'N/A')}")
    print(f"   Amount: ${tx.get('Amount', 0):,.2f}")
    print(f"   Merchant: {tx.get('Merchant', 'N/A')}")
    print(f"   Country: {tx.get('Country', 'N/A')}")
    print(f"   Risk Score: {tx['risk_score']:.1f}/100 ({tx['risk_level']})")
    print(f"   Flags: {', '.join(tx['risk_flags'])}")

print("\n" + "=" * 80)

# Export
analyzed_df = report['analyzed_data']
analyzed_df.to_csv('account_statement_analyzed.csv', index=False)
print(f"💾 Analysis saved to: account_statement_analyzed.csv")
