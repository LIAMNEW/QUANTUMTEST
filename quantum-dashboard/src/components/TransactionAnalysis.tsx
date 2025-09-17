import React, { useState } from 'react';

interface TransactionData {
  topSenders: Array<{
    address: string;
    value: number;
    isWatchlisted: boolean;
  }>;
  topReceivers: Array<{
    address: string;
    value: number;
    isWatchlisted: boolean;
  }>;
}

interface TransactionAnalysisProps {
  transactions: TransactionData;
}

const TransactionAnalysis: React.FC<TransactionAnalysisProps> = ({ transactions }) => {
  const [viewMode, setViewMode] = useState<'senders' | 'receivers'>('senders');
  
  const currentData = viewMode === 'senders' ? transactions?.topSenders : transactions?.topReceivers;
  const maxValue = Math.max(...(currentData?.map(item => item.value) || [1]));

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatValue = (value: number) => {
    return `$${(value / 1000).toFixed(1)}K`;
  };

  return (
    <div className=\"bg-dark-card border border-dark-border rounded-2xl p-6 hover:shadow-neon-green/20 transition-all duration-300\">
      {/* Header */}
      <div className=\"flex items-center justify-between mb-6\">
        <div>
          <h2 className=\"text-xl font-bold text-white flex items-center gap-2\">
            <span className=\"text-neon-green\">📊</span>
            Transaction Analysis
          </h2>
          <p className=\"text-gray-400 text-sm mt-1\">Network activity overview</p>
        </div>

        {/* Toggle Buttons */}
        <div className=\"flex bg-dark-surface rounded-xl p-1 border border-dark-border\">
          <button
            onClick={() => setViewMode('senders')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${\n              viewMode === 'senders'\n                ? 'bg-neon-green text-black shadow-neon-green/50'\n                : 'text-gray-400 hover:text-white'\n            }`}
          >
            Top Senders
          </button>\n          <button\n            onClick={() => setViewMode('receivers')}\n            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${\n              viewMode === 'receivers'\n                ? 'bg-neon-blue text-black shadow-neon-blue/50'\n                : 'text-gray-400 hover:text-white'\n            }`}\n          >\n            Top Receivers\n          </button>\n        </div>\n      </div>\n\n      {/* Chart */}\n      <div className=\"space-y-4\">\n        {currentData?.map((item, index) => {\n          const percentage = (item.value / maxValue) * 100;\n          const barColor = viewMode === 'senders' ? 'neon-green' : 'neon-blue';\n          \n          return (\n            <div key={index} className=\"group\">\n              <div className=\"flex items-center justify-between mb-2\">\n                <div className=\"flex items-center gap-3\">\n                  <span className=\"text-gray-400 text-sm w-4\">{index + 1}</span>\n                  <div className=\"flex items-center gap-2\">\n                    {item.isWatchlisted && (\n                      <span className=\"text-neon-yellow animate-pulse\" title=\"Watchlisted\">🏷️</span>\n                    )}\n                    <span className=\"font-mono text-white text-sm\">\n                      {formatAddress(item.address)}\n                    </span>\n                  </div>\n                </div>\n                <span className={`font-bold text-${barColor} text-sm`}>\n                  {formatValue(item.value)}\n                </span>\n              </div>\n              \n              {/* Animated Bar */}\n              <div className=\"relative h-8 bg-dark-surface rounded-full overflow-hidden border border-dark-border\">\n                <div \n                  className={`absolute left-0 top-0 h-full bg-gradient-to-r ${\n                    viewMode === 'senders' \n                      ? 'from-neon-green/80 to-neon-green/60' \n                      : 'from-neon-blue/80 to-neon-blue/60'\n                  } rounded-full transition-all duration-1000 ease-out group-hover:shadow-${barColor}/50`}\n                  style={{ width: `${percentage}%` }}\n                >\n                  {/* Animated shine effect */}\n                  <div className=\"absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse\"></div>\n                </div>\n                \n                {/* Risk indicator for watchlisted addresses */}\n                {item.isWatchlisted && (\n                  <div className=\"absolute right-2 top-1/2 transform -translate-y-1/2\">\n                    <div className=\"w-2 h-2 bg-neon-yellow rounded-full animate-ping\"></div>\n                  </div>\n                )}\n              </div>\n            </div>\n          );\n        })}\n      </div>\n\n      {/* Legend */}\n      <div className=\"flex items-center justify-between mt-6 pt-4 border-t border-dark-border\">\n        <div className=\"flex items-center gap-4 text-xs\">\n          <div className=\"flex items-center gap-2\">\n            <div className={`w-3 h-3 rounded bg-${viewMode === 'senders' ? 'neon-green' : 'neon-blue'}`}></div>\n            <span className=\"text-gray-400\">{viewMode === 'senders' ? 'Outbound' : 'Inbound'} Volume</span>\n          </div>\n          <div className=\"flex items-center gap-2\">\n            <span className=\"text-neon-yellow\">🏷️</span>\n            <span className=\"text-gray-400\">Watchlisted</span>\n          </div>\n        </div>\n        <div className=\"text-xs text-gray-500\">\n          Updated: {new Date().toLocaleTimeString()}\n        </div>\n      </div>\n    </div>\n  );\n};\n\nexport default TransactionAnalysis;