import React from 'react';

interface Alert {
  id: number;
  type: string;
  address: string;
  reason: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

interface AlertsFeedProps {
  alerts: Alert[];
}

const AlertsFeed: React.FC<AlertsFeedProps> = ({ alerts }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-400';
      case 'high':
        return 'text-orange-400';
      case 'medium':
        return 'text-neon-yellow';
      case 'low':
        return 'text-neon-green';
      default:
        return 'text-gray-400';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '🚨';
      case 'high':
        return '⚠️';
      case 'medium':
        return '🟡';
      case 'low':
        return '🟢';
      default:
        return '🔘';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'high_risk':
        return '🎨';
      case 'anomaly':
        return '🔍';
      case 'compliance':
        return '📝';
      case 'suspicious':
        return '👀';
      default:
        return '📢';
    }
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  return (
    <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:shadow-neon-yellow/20 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-neon-yellow">📢</span>
            Live Alerts
          </h2>
          <p className="text-gray-400 text-sm mt-1">Real-time security notifications</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-neon-yellow rounded-full animate-ping"></div>
          <span className="text-neon-yellow text-xs font-medium">Live</span>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3 max-h-80 overflow-y-auto custom-scrollbar">
        {alerts?.map((alert, index) => (
          <div
            key={alert.id}
            className="group bg-dark-surface border border-dark-border rounded-xl p-4 hover:border-neon-yellow/50 transition-all duration-300 cursor-pointer"
          >
            {/* Alert Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <span className="text-lg">{getTypeIcon(alert.type)}</span>
                  <span className="text-sm">{getSeverityIcon(alert.severity)}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-white text-sm">
                      {formatAddress(alert.address)}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${
                      alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                      alert.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                      alert.severity === 'medium' ? 'bg-yellow-500/20 text-neon-yellow' :
                      'bg-green-500/20 text-neon-green'
                    }`}>
                      {alert.severity}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    {alert.reason}
                  </p>
                </div>
              </div>
              <div className="text-xs text-gray-400 ml-4">
                {formatTime(alert.timestamp)}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button className="px-3 py-1 bg-neon-blue/20 text-neon-blue hover:bg-neon-blue hover:text-black rounded-lg text-xs font-medium transition-all duration-200">
                  Investigate
                </button>
                <button className="px-3 py-1 bg-dark-border text-gray-400 hover:text-white rounded-lg text-xs font-medium transition-all duration-200">
                  Dismiss
                </button>
              </div>
              <div className="text-xs text-gray-500">
                #{alert.id.toString().padStart(4, '0')}
              </div>
            </div>

            {/* Severity Indicator Bar */}
            <div className="mt-3 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className={`h-full transition-all duration-300 group-hover:animate-pulse ${
                alert.severity === 'critical' ? 'bg-red-400' :
                alert.severity === 'high' ? 'bg-orange-400' :
                alert.severity === 'medium' ? 'bg-neon-yellow' :
                'bg-neon-green'
              }`} style={{ width: '100%' }}></div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-dark-border">
        <div className="text-xs text-gray-400">
          {alerts?.length || 0} active alerts
        </div>
        <button className="text-neon-yellow hover:text-white text-xs font-medium transition-colors">
          View All Alerts →
        </button>
      </div>
    </div>
  );
};

export default AlertsFeed;