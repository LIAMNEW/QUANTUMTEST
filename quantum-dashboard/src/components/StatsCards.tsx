import React from 'react';

interface StatsCardsProps {
  stats: {
    suspiciousTransactions: number;
    highRiskWallets: number;
    aiAlerts: number;
    complianceStatus: string;
  };
}

const StatsCards: React.FC<StatsCardsProps> = ({ stats }) => {
  const cards = [
    {
      title: 'Suspicious Transactions',
      value: stats?.suspiciousTransactions || 0,
      icon: '⚠️',
      color: 'neon-yellow',
      bgGradient: 'from-yellow-500/20 to-orange-500/20'
    },
    {
      title: 'High-Risk Wallets',
      value: stats?.highRiskWallets || 0,
      icon: '🔍',
      color: 'neon-purple',
      bgGradient: 'from-purple-500/20 to-pink-500/20'
    },
    {
      title: 'AI Alerts',
      value: stats?.aiAlerts || 0,
      icon: '🤖',
      color: 'neon-blue',
      bgGradient: 'from-blue-500/20 to-cyan-500/20'
    },
    {
      title: 'Compliance Status',
      value: stats?.complianceStatus || 'Unknown',
      icon: '✅',
      color: 'neon-green',
      bgGradient: 'from-green-500/20 to-emerald-500/20'
    }
  ];

  return (
    <div className=\"grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6\">
      {cards.map((card, index) => (
        <div
          key={index}
          className={`bg-gradient-to-br ${card.bgGradient} backdrop-blur-sm bg-dark-card border border-dark-border rounded-2xl p-6 hover:shadow-${card.color} transition-all duration-300 hover:scale-105 group cursor-pointer`}
        >
          <div className=\"flex items-center justify-between mb-4\">
            <div className=\"text-2xl group-hover:animate-bounce\">{card.icon}</div>
            <div className={`w-3 h-3 rounded-full bg-${card.color} animate-pulse`}></div>
          </div>
          
          <div className=\"space-y-2\">
            <p className=\"text-gray-400 text-sm font-medium\">{card.title}</p>
            <div className=\"flex items-baseline gap-2\">
              <span className={`text-2xl font-bold text-${card.color} group-hover:animate-glow`}>
                {typeof card.value === 'number' ? card.value.toLocaleString() : card.value}
              </span>
              {typeof card.value === 'number' && (
                <span className=\"text-xs text-gray-500\">
                  {index === 0 && '+12%'}
                  {index === 1 && '+5%'}
                  {index === 2 && '+8%'}
                </span>
              )}
            </div>
          </div>

          {/* Neon border effect */}
          <div className={`absolute inset-0 rounded-2xl border-2 border-${card.color} opacity-0 group-hover:opacity-30 transition-opacity duration-300`}></div>
        </div>
      ))}
    </div>
  );
};

export default StatsCards;