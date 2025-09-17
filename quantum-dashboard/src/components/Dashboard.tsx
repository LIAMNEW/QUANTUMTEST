import React, { useState, useEffect } from 'react';
import StatsCards from './StatsCards';
import TransactionAnalysis from './TransactionAnalysis';
import RiskAssessment from './RiskAssessment';
import SecurityPlans from './SecurityPlans';
import AlertsFeed from './AlertsFeed';
import QuickActions from './QuickActions';

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate fetching data from the existing Streamlit backend
    const fetchData = async () => {
      try {
        // This would connect to your existing backend
        // For now, we'll use sample data structure that matches your existing data
        const sampleData = {
          stats: {
            suspiciousTransactions: 47,
            highRiskWallets: 12,
            aiAlerts: 8,
            complianceStatus: 'Compliant'
          },
          transactions: {
            topSenders: [
              { address: '0x1234...5678', value: 150000, isWatchlisted: false },
              { address: '0xabcd...efgh', value: 120000, isWatchlisted: true },
              { address: '0x9999...1111', value: 98000, isWatchlisted: false }
            ],
            topReceivers: [
              { address: '0x2468...1357', value: 180000, isWatchlisted: true },
              { address: '0xffff...aaaa', value: 145000, isWatchlisted: false },
              { address: '0x7777...3333', value: 110000, isWatchlisted: false }
            ]
          },
          riskAssessment: {
            low: 45,
            medium: 32,
            high: 18,
            critical: 5
          },
          securityPlans: [
            { name: 'Exchange Investigation', progress: 75, status: 'active' },
            { name: 'Mixer Analysis', progress: 40, status: 'pending' },
            { name: 'DeFi Protocol Review', progress: 90, status: 'completed' }
          ],
          alerts: [
            {
              id: 1,
              type: 'high_risk',
              address: '0xabcd...efgh',
              reason: 'Large volume transactions detected',
              timestamp: '2025-01-17 10:30:00',
              severity: 'high'
            },
            {
              id: 2,
              type: 'anomaly',
              address: '0x2468...1357',
              reason: 'Unusual transaction pattern',
              timestamp: '2025-01-17 09:45:00',
              severity: 'medium'
            }
          ]
        };
        
        setDashboardData(sampleData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className=\"min-h-screen bg-dark-bg flex items-center justify-center\">
        <div className=\"animate-pulse-neon\">
          <div className=\"w-16 h-16 border-4 border-neon-green border-t-transparent rounded-full animate-spin\"></div>
        </div>
      </div>
    );
  }

  return (
    <div className=\"min-h-screen bg-dark-bg p-6\">
      <div className=\"max-w-7xl mx-auto space-y-6\">
        {/* Header */}
        <header className=\"flex items-center justify-between mb-8\">
          <div>
            <h1 className=\"text-3xl font-bold text-white flex items-center gap-3\">
              <span className=\"text-neon-green\">🛡️</span>
              QuantumGuard AI
              <span className=\"text-neon-blue animate-glow\">⚡</span>
            </h1>
            <p className=\"text-gray-400 mt-1\">Advanced Fraud Detection Dashboard</p>
          </div>
          <div className=\"flex items-center gap-4\">
            <div className=\"px-3 py-1 bg-dark-card border border-neon-green rounded-full text-sm\">
              <span className=\"text-neon-green\">●</span> Live Monitoring
            </div>
          </div>
        </header>

        {/* Stats Cards Row */}
        <StatsCards stats={dashboardData?.stats} />

        {/* Main Content Grid */}
        <div className=\"grid grid-cols-1 xl:grid-cols-3 gap-6\">
          {/* Left Column - Transaction Analysis & Risk Assessment */}
          <div className=\"xl:col-span-2 space-y-6\">
            <TransactionAnalysis transactions={dashboardData?.transactions} />
            <RiskAssessment riskData={dashboardData?.riskAssessment} />
          </div>

          {/* Right Column - Security Plans, Alerts & Quick Actions */}
          <div className=\"space-y-6\">
            <QuickActions />
            <SecurityPlans plans={dashboardData?.securityPlans} />
            <AlertsFeed alerts={dashboardData?.alerts} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;