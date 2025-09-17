import React from 'react';

const QuickActions: React.FC = () => {
  const actions = [
    {
      name: 'Run Analysis',
      icon: '🚀',
      color: 'neon-green',
      description: 'Start new scan'
    },
    {
      name: 'Export Report',
      icon: '📊',
      color: 'neon-blue',
      description: 'Generate PDF'
    },
    {
      name: 'Upload Data',
      icon: '📤',
      color: 'neon-yellow',
      description: 'Import transactions'
    },
    {
      name: 'AI Search',
      icon: '🤖',
      color: 'neon-purple',
      description: 'Query patterns'
    },
    {
      name: 'Live Monitor',
      icon: '📡',
      color: 'neon-green',
      description: 'Real-time feed'
    },
    {
      name: 'Settings',
      icon: '⚙️',
      color: 'gray-400',
      description: 'Configure alerts'
    }
  ];

  return (
    <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:shadow-neon-green/20 transition-all duration-300">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="text-neon-green">⚡</span>
          Quick Actions
        </h2>
        <p className="text-gray-400 text-sm mt-1">Essential tools & shortcuts</p>
      </div>

      {/* Actions Grid */}
      <div className="grid grid-cols-2 gap-3">
        {actions.map((action, index) => (
          <button
            key={index}
            className={`group bg-dark-surface border border-dark-border rounded-xl p-4 hover:border-${action.color}/50 hover:shadow-${action.color}/20 transition-all duration-300 text-left`}
          >
            {/* Action Icon & Name */}
            <div className="flex items-start gap-3 mb-2">
              <div className="text-2xl group-hover:animate-bounce">
                {action.icon}
              </div>
              <div className="flex-1">
                <h3 className={`text-white font-medium text-sm group-hover:text-${action.color} transition-colors`}>
                  {action.name}
                </h3>
                <p className="text-gray-400 text-xs mt-1">
                  {action.description}
                </p>
              </div>
            </div>

            {/* Action Status/Indicator */}
            <div className="flex items-center justify-between">
              <div className={`w-2 h-2 rounded-full bg-${action.color} opacity-60 group-hover:opacity-100 transition-opacity`}></div>
              <div className="text-gray-500 group-hover:text-gray-300 transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="mt-6 pt-4 border-t border-dark-border">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-dark-surface rounded-lg p-3">
            <div className="text-neon-green font-bold text-lg">24/7</div>
            <div className="text-gray-400 text-xs">Monitoring</div>
          </div>
          <div className="bg-dark-surface rounded-lg p-3">
            <div className="text-neon-blue font-bold text-lg">99.9%</div>
            <div className="text-gray-400 text-xs">Uptime</div>
          </div>
          <div className="bg-dark-surface rounded-lg p-3">
            <div className="text-neon-yellow font-bold text-lg">&lt;1s</div>
            <div className="text-gray-400 text-xs">Response</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuickActions;