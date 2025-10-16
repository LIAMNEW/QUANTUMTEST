"""
Test AUSTRAC risk calculator directly
"""

import pandas as pd
from austrac_risk_calculator import calculate_austrac_risk_score

# Load the account statement
df = pd.read_csv('attached_assets/account_statement_100_1760573773832.csv')

print("🔍 Testing AUSTRAC Risk Calculator\n")
print("=" * 80)
print(f"Columns: {list(df.columns)}")
print(f"Total Transactions: {len(df)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print("=" * 80)

# Calculate risk score
print("\n📊 Calculating AUSTRAC compliance risk score...\n")
risk_data = calculate_austrac_risk_score(df)

print("=" * 80)
print("AUSTRAC RISK SCORE RESULTS")
print("=" * 80)

print(f"\n🎯 Risk Percentage: {risk_data['risk_percentage']}%")
print(f"📊 Risk Level: {risk_data['risk_level']}")
print(f"🚨 Risk Status: {risk_data['risk_status']}")
print(f"\n📈 Transactions Analyzed: {risk_data['transactions_analyzed']}")
print(f"🔴 High Risk Count: {risk_data['high_risk_count']}")
print(f"⚠️  Critical Count: {risk_data['critical_count']}")
print(f"📋 Reporting Required: {risk_data['reporting_required']}")

print(f"\n📊 Individual Risk Scores:")
print(f"   Average: {risk_data['avg_individual_risk']}")
print(f"   Maximum: {risk_data['max_individual_risk']}")

print("\n" + "=" * 80)
print(risk_data['summary_message'])
print("=" * 80)

print("\n📋 Compliance Recommendations:")
for rec in risk_data['compliance_recommendations']:
    print(f"  • {rec}")
