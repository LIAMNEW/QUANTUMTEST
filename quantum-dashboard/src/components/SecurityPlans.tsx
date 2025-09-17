import React from 'react';

interface SecurityPlan {
  name: string;
  progress: number;
  status: 'active' | 'pending' | 'completed';
}

interface SecurityPlansProps {
  plans: SecurityPlan[];
}

const SecurityPlans: React.FC<SecurityPlansProps> = ({ plans }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'neon-green';
      case 'active':
        return 'neon-blue';
      case 'pending':
        return 'neon-yellow';
      default:
        return 'gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'active':
        return '🔄';
      case 'pending':
        return '⏳';
      default:
        return '❓';
    }
  };

  return (
    <div className="bg-dark-card border border-dark-border rounded-2xl p-6 hover:shadow-neon-purple/20 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-neon-purple">🛡️</span>
            Security Plans
          </h2>
          <p className="text-gray-400 text-sm mt-1">Investigation progress</p>
        </div>
        <button className="text-neon-purple hover:text-white transition-colors text-sm">
          View All
        </button>
      </div>

      {/* Plans List */}
      <div className="space-y-4">
        {plans?.map((plan, index) => (
          <div
            key={index}
            className="group bg-dark-surface border border-dark-border rounded-xl p-4 hover:border-neon-purple/50 transition-all duration-300"
          >
            {/* Plan Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className="text-lg">{getStatusIcon(plan.status)}</span>
                <div>
                  <h3 className="text-white font-medium text-sm">{plan.name}</h3>
                  <p className="text-gray-400 text-xs capitalize">{plan.status}</p>
                </div>
              </div>
              <div className={`text-${getStatusColor(plan.status)} font-bold text-sm`}>
                {plan.progress}%
              </div>
            </div>

            {/* Progress Bar */}
            <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden mb-3">
              <div
                className={`absolute left-0 top-0 h-full bg-gradient-to-r ${
                  plan.status === 'completed'
                    ? 'from-neon-green/80 to-neon-green/60'
                    : plan.status === 'active'
                    ? 'from-neon-blue/80 to-neon-blue/60'
                    : 'from-neon-yellow/80 to-neon-yellow/60'
                } rounded-full transition-all duration-1000 ease-out`}
                style={{ width: `${plan.progress}%` }}
              >
                {/* Animated shine effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse"></div>
              </div>
            </div>

            {/* Action Button */}
            <div className="flex justify-end">
              <button
                className={`px-4 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
                  plan.status === 'completed'
                    ? 'bg-dark-border text-gray-400 cursor-not-allowed'
                    : 'bg-neon-purple/20 text-neon-purple hover:bg-neon-purple hover:text-black border border-neon-purple/50 hover:shadow-neon-purple/50'
                }`}
                disabled={plan.status === 'completed'}
              >
                {plan.status === 'completed' ? 'Completed' : 'Investigate'}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Add New Plan Button */}
      <button className="w-full mt-4 py-3 border-2 border-dashed border-dark-border hover:border-neon-purple/50 rounded-xl text-gray-400 hover:text-neon-purple transition-all duration-300 flex items-center justify-center gap-2">
        <span>➕</span>
        <span className="text-sm font-medium">Add New Investigation</span>
      </button>
    </div>
  );
};

export default SecurityPlans;