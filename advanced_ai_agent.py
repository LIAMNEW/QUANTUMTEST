"""
Advanced Agentic AI Assistant for QuantumGuard AI
Provides intelligent application control and assistance
"""

import os
from openai import OpenAI
import json
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class AdvancedAIAgent:
    """
    Advanced agentic AI assistant that can control the app and provide assistance
    """
    
    def __init__(self):
        self.conversation_history = []
        self.system_context = self._build_system_context()
        
    def _build_system_context(self):
        """Build comprehensive system context for the AI agent"""
        return """You are an advanced AI assistant for QuantumGuard AI, a comprehensive blockchain transaction analysis platform.

Your capabilities and knowledge:
- **Application Control**: You can guide users through the app features, explain settings, and recommend configurations
- **Data Analysis**: Expert in blockchain transactions, risk assessment, and anomaly detection
- **Security Expertise**: Knowledge of quantum cryptography, enterprise security, and AUSTRAC compliance
- **Machine Learning**: Understanding of LSTM autoencoders, VAE, GNNs, and ensemble methods
- **User Assistance**: Provide step-by-step guidance and troubleshooting

Application Features You Can Help With:
1. **Data Upload**: CSV files, Blockchain APIs (Bitcoin, Ethereum, Coinbase, Binance)
2. **Analysis Configuration**: Risk thresholds, anomaly sensitivity settings
3. **Advanced ML Models**: Enhanced anomaly detection with GPT-5, LSTM, VAE, Graph Neural Networks
4. **Visualizations**: Network graphs, risk heatmaps, anomaly detection plots, timelines
5. **Security Features**: Quantum cryptography, MFA, backup/recovery, enterprise key management
6. **AUSTRAC Compliance**: Transaction monitoring, reporting codes, risk scoring
7. **AI Analytics**: Predictive analysis, behavioral patterns, multimodal insights

Communication Guidelines:
- Be concise, clear, and actionable
- Provide step-by-step instructions when needed
- Suggest best practices and optimal configurations
- Explain technical concepts in simple terms
- Offer proactive recommendations based on user's context
- When uncertain, acknowledge limitations and suggest alternatives
- IMPORTANT: Never use HTML tags or markup in your responses - always use plain text only
- Use markdown formatting (**, *, lists) for emphasis if needed, but never HTML

Always aim to be helpful, accurate, and user-focused."""

    def chat(self, user_message, app_context=None):
        """
        Main chat interface for the AI agent
        
        Args:
            user_message: User's question or request
            app_context: Optional context about current app state
            
        Returns:
            AI assistant's response
        """
        try:
            # Build messages with conversation history
            messages = [
                {"role": "system", "content": self.system_context}
            ]
            
            # Add app context if provided
            if app_context:
                context_message = f"Current app context: {json.dumps(app_context, indent=2)}"
                messages.append({"role": "system", "content": context_message})
            
            # Add conversation history
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages
                messages.append(msg)
            
            # Add user message
            messages.append({"role": "user", "content": user_message})
            
            # Call GPT-4o for high-quality responses
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for reliable, quality responses
                messages=messages,
                max_completion_tokens=800,
                temperature=0.7  # Balanced creativity and consistency
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
            print(f"Advanced AI Agent Error: {str(e)}")
            return error_msg
    
    def get_quick_action_suggestions(self, app_state):
        """
        Provide quick action suggestions based on current app state
        
        Args:
            app_state: Dictionary containing current app state information
            
        Returns:
            List of suggested actions
        """
        suggestions = []
        
        if not app_state.get('has_data'):
            suggestions.append("📁 Upload transaction data to begin analysis")
            suggestions.append("🔗 Connect to blockchain APIs for live data")
        
        if app_state.get('has_data') and not app_state.get('analysis_complete'):
            suggestions.append("🚀 Run complete blockchain analysis")
            suggestions.append("⚙️ Configure risk thresholds and anomaly sensitivity")
        
        if app_state.get('analysis_complete'):
            suggestions.append("🔍 Explore AI-powered insights and visualizations")
            suggestions.append("💾 Save analysis session for future reference")
            suggestions.append("📊 Generate PDF report of findings")
        
        if app_state.get('high_risk_found'):
            suggestions.append("🚨 Investigate high-risk transactions immediately")
            suggestions.append("🏷️ Add suspicious addresses to watchlist")
        
        if app_state.get('anomalies_found'):
            suggestions.append("🔎 Review detected anomalies in detail")
            suggestions.append("🧠 Use AI assistant to analyze anomaly patterns")
        
        return suggestions
    
    def explain_feature(self, feature_name):
        """
        Provide detailed explanation of a specific feature
        
        Args:
            feature_name: Name of the feature to explain
            
        Returns:
            Detailed explanation
        """
        explanations = {
            "quantum_security": "Post-quantum cryptography protects your data against future quantum computing threats using AES-256-GCM encryption with PBKDF2 key derivation (480,000 iterations). Your financial information remains secure even when quantum computers become powerful enough to break traditional encryption.",
            
            "enhanced_ml": "Enhanced ML uses cutting-edge algorithms: LSTM Autoencoders analyze transaction sequences, Variational Autoencoders provide probabilistic modeling, Graph Neural Networks examine network relationships, and Ensemble Methods combine multiple algorithms for 95%+ accuracy improvement.",
            
            "austrac_compliance": "AUSTRAC compliance features help Australian financial institutions meet regulatory requirements. The system automatically classifies transactions, calculates risk scores, and provides reporting codes aligned with AUSTRAC guidelines.",
            
            "anomaly_detection": "Anomaly detection identifies unusual transaction patterns that may indicate fraud or suspicious activity. The system uses ensemble methods combining Isolation Forest, LSTM autoencoders, and VAEs for superior accuracy.",
            
            "risk_assessment": "Risk assessment scores transactions based on multiple factors: amount, frequency, network patterns, and behavioral analysis. Scores range from 0 (low risk) to 1 (critical risk), helping prioritize investigations.",
            
            "ai_insights": "AI insights use GPT-5 to analyze your transaction data and provide natural language explanations of patterns, risks, and recommendations. Simply ask questions about your data and get intelligent responses."
        }
        
        return explanations.get(feature_name.lower(), 
            f"I don't have a detailed explanation for '{feature_name}' yet. Please ask me specific questions about this feature!")
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        return "Conversation history cleared. How can I help you?"


# Global instance
advanced_agent = AdvancedAIAgent()


def get_agent_response(user_message, app_context=None):
    """
    Convenience function to get agent response
    
    Args:
        user_message: User's question
        app_context: Optional app state context
        
    Returns:
        AI response
    """
    return advanced_agent.chat(user_message, app_context)


def get_quick_suggestions(app_state):
    """Get quick action suggestions"""
    return advanced_agent.get_quick_action_suggestions(app_state)


def explain_app_feature(feature_name):
    """Get feature explanation"""
    return advanced_agent.explain_feature(feature_name)
