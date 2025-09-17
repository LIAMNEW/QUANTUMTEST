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
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 hover:shadow-green-500/20 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-green-500">📊</span>
            Transaction Analysis
          </h2>
          <p className="text-gray-400 text-sm mt-1">Network activity overview</p>
        </div>

        {/* Toggle Buttons */}
        <div className="flex bg-gray-800 rounded-xl p-1 border border-gray-700">
          <button
            onClick={() => setViewMode('senders')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              viewMode === 'senders'
                ? 'bg-green-500 text-black shadow-green-500/50'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Top Senders
          </button>
          <button
            onClick={() => setViewMode('receivers')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              viewMode === 'receivers'
                ? 'bg-green-500 text-black shadow-green-500/50'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Top Receivers
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="space-y-4">
        {currentData?.map((item, index) => {
          const percentage = (item.value / maxValue) * 100;
          
          return (
            <div key={index} className="group">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className="text-gray-400 text-sm w-4">{index + 1}</span>
                  <div className="flex items-center gap-2">
                    {item.isWatchlisted && (
                      <span className="text-green-400 animate-pulse" title="Watchlisted">🏷️</span>
                    )}
                    <span className="font-mono text-white text-sm">
                      {formatAddress(item.address)}
                    </span>
                  </div>
                </div>
                <span className="font-bold text-green-500 text-sm">
                  {formatValue(item.value)}
                </span>
              </div>
              
              {/* Animated Bar */}
              <div className="relative h-8 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
                <div 
                  className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500/80 to-green-400/60 rounded-full transition-all duration-1000 ease-out group-hover:shadow-green-500/50"
                  style={{ width: `${percentage}%` }}
                >
                  {/* Animated shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-gray-700">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-500 font-bold text-lg">{currentData?.length || 0}</div>
          <div className="text-gray-400 text-xs">Addresses</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-500 font-bold text-lg">
            {currentData?.filter(item => item.isWatchlisted).length || 0}
          </div>
          <div className="text-gray-400 text-xs">Watchlisted</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-500 font-bold text-lg">
            {currentData ? formatValue(currentData.reduce((sum, item) => sum + item.value, 0)) : '$0'}
          </div>
          <div className="text-gray-400 text-xs">Total Volume</div>
        </div>
      </div>
    </div>
  );
};

export default TransactionAnalysis;