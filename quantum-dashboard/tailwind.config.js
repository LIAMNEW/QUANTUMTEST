/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neon-green': '#00ff88',
        'neon-blue': '#00d4ff',
        'neon-yellow': '#ffff00',
        'neon-purple': '#bf00ff',
        'dark-bg': '#0a0a0f',
        'dark-card': '#1a1a2e',
        'dark-surface': '#16213e',
        'dark-border': '#2a2d47',
      },
      boxShadow: {
        'neon-green': '0 0 20px rgba(0, 255, 136, 0.3)',
        'neon-blue': '0 0 20px rgba(0, 212, 255, 0.3)',
        'neon-yellow': '0 0 20px rgba(255, 255, 0, 0.3)',
        'neon-purple': '0 0 20px rgba(191, 0, 255, 0.3)',
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': {
            opacity: '1',
            boxShadow: '0 0 20px rgba(0, 255, 136, 0.3)',
          },
          '50%': {
            opacity: '0.8',
            boxShadow: '0 0 30px rgba(0, 255, 136, 0.6)',
          },
        },
        'glow': {
          'from': {
            boxShadow: '0 0 20px rgba(0, 255, 136, 0.2)',
          },
          'to': {
            boxShadow: '0 0 30px rgba(0, 255, 136, 0.4)',
          },
        },
      },
    },
  },
  plugins: [],
}

