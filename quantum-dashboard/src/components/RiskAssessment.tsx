import React from 'react';

interface RiskData {
  low: number;
  medium: number;
  high: number;
  critical: number;
}

interface RiskAssessmentProps {
  riskData: RiskData;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ riskData }) => {
  const total = (riskData?.low || 0) + (riskData?.medium || 0) + (riskData?.high || 0) + (riskData?.critical || 0);
  
  const riskLevels = [
    {
      level: 'Low Risk',
      count: riskData?.low || 0,
      percentage: total > 0 ? ((riskData?.low || 0) / total * 100) : 0,
      color: 'green-500',
      bgColor: 'bg-green-500/40',
      icon: '●'
    },
    {
      level: 'Medium Risk',
      count: riskData?.medium || 0,
      percentage: total > 0 ? ((riskData?.medium || 0) / total * 100) : 0,
      color: 'gray-400',
      bgColor: 'bg-gray-500/40',
      icon: '●'
    },
    {
      level: 'High Risk',
      count: riskData?.high || 0,
      percentage: total > 0 ? ((riskData?.high || 0) / total * 100) : 0,
      color: 'gray-600',
      bgColor: 'bg-gray-700/40',
      icon: '●'
    },
    {
      level: 'Critical Risk',
      count: riskData?.critical || 0,
      percentage: total > 0 ? ((riskData?.critical || 0) / total * 100) : 0,
      color: 'black',
      bgColor: 'bg-black/60',
      icon: '●'
    }
  ];

  const highRiskPercentage = total > 0 ? (((riskData?.high || 0) + (riskData?.critical || 0)) / total * 100) : 0;
  const riskScore = total > 0 ? (
    ((riskData?.low || 0) * 1 + (riskData?.medium || 0) * 2 + (riskData?.high || 0) * 3 + (riskData?.critical || 0) * 4) / total / 4 * 100
  ) : 0;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 hover:shadow-green-500/20 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-green-500">⚡</span>
            Risk Assessment
          </h2>
          <p className="text-gray-400 text-sm mt-1">Transaction risk distribution</p>
        </div>
        
        {/* Overall Risk Score */}
        <div className="text-center">
          <div className={`text-2xl font-bold ${
            riskScore < 25 ? 'text-green-500' :
            riskScore < 50 ? 'text-gray-400' :
            riskScore < 75 ? 'text-gray-600' : 'text-black bg-white rounded px-2'
          }`}>
            {riskScore.toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Risk Score</div>
        </div>
      </div>

      {/* Progress Bars */}
      <div className="space-y-4 mb-6">
        {riskLevels.map((risk, index) => (
          <div key={index} className="group">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className={`text-sm text-${risk.color}`}>{risk.icon}</span>
                <span className="text-white text-sm font-medium">{risk.level}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-${risk.color} font-bold text-sm`}>
                  {risk.count.toLocaleString()}
                </span>
                <span className="text-gray-400 text-xs">
                  ({risk.percentage.toFixed(1)}%)
                </span>
              </div>
            </div>
            
            {/* Animated Progress Bar */}
            <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
              <div 
                className={`absolute left-0 top-0 h-full ${risk.bgColor} rounded-full transition-all duration-1000 ease-out group-hover:shadow-lg`}
                style={{ width: `${risk.percentage}%` }}
              >
                {/* Animated shine effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Distribution Chart */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
          <span className="text-green-500">📊</span>
          Risk Distribution
        </h3>
        
        {/* Horizontal Stacked Bar */}
        <div className="relative h-8 bg-black rounded-full overflow-hidden mb-3 border border-gray-600">
          {riskLevels.map((risk, index) => {
            const leftOffset = riskLevels.slice(0, index).reduce((sum, r) => sum + r.percentage, 0);
            return (
              risk.percentage > 0 && (
                <div
                  key={index}
                  className={`absolute top-0 h-full ${risk.bgColor} transition-all duration-1000 ease-out hover:brightness-125`}
                  style={{
                    left: `${leftOffset}%`,
                    width: `${risk.percentage}%`
                  }}
                  title={`${risk.level}: ${risk.count} (${risk.percentage.toFixed(1)}%)`}
                >
                  {risk.percentage > 10 && (
                    <div className="flex items-center justify-center h-full text-xs font-medium text-white">
                      {risk.percentage.toFixed(0)}%
                    </div>
                  )}
                </div>
              )
            );
          })}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
            <div className="text-gray-400 mb-1">High Risk+</div>
            <div className={`font-bold ${
              highRiskPercentage > 20 ? 'text-black bg-white rounded px-2' :
              highRiskPercentage > 10 ? 'text-gray-400' : 'text-green-500'
            }`}>
              {highRiskPercentage.toFixed(1)}%
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-700">
            <div className="text-gray-400 mb-1">Total Analyzed</div>
            <div className="text-white font-bold">
              {total.toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-700 text-xs">
        <div className="text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${
          highRiskPercentage > 20 ? 'bg-black/80 text-white border border-white' :
          highRiskPercentage > 10 ? 'bg-gray-700/80 text-gray-300' :
          'bg-green-500/20 text-green-500'
        }`}>
          {highRiskPercentage > 20 ? 'High Alert' :
           highRiskPercentage > 10 ? 'Monitor' : 'Normal'}
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;