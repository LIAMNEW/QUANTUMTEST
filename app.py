import streamlit as st
import os
from openai import OpenAI

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Dark theme styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Main content area */
    .main {
        background-color: #0e1117;
    }
    
    /* Chat container */
    .chat-container {
        background: linear-gradient(135deg, #1a1d29 0%, #262b3d 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #1a1d29;
        color: white;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Header styling */
    h1 {
        color: #e2e8f0;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h3 {
        color: #cbd5e0;
        text-align: center;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #262b3d 100%);
    }
    
    /* Clear button */
    .clear-button button {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    /* Message container clearfix */
    .message-container::after {
        content: "";
        display: table;
        clear: both;
    }
</style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

st.markdown("<h1>🤖 AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3>Powered by GPT-5 | Advanced Conversational AI</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col3:
    if st.button("🗑️ Clear Conversation", key="clear_btn", help="Start a new conversation"):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'''
        <div class="message-container">
            <div class="user-message">
                <strong>You</strong><br>
                {content}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="message-container">
            <div class="assistant-message">
                <strong>🤖 AI Assistant</strong><br>
                {content}
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_input, col_button = st.columns([4, 1])

with col_input:
    user_input = st.text_input(
        "Message",
        placeholder="How can I reduce my risk score?",
        key="user_input",
        label_visibility="collapsed"
    )

with col_button:
    send_button = st.button("📤 Send", type="primary", use_container_width=True)

if send_button and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_started = True
    
    with st.spinner("🤔 Thinking..."):
        try:
            system_message = """You are an advanced AI assistant for QuantumGuard AI, a comprehensive blockchain transaction analysis platform.

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

Always aim to be helpful, accurate, and user-focused."""

            messages_for_api = [{"role": "system", "content": system_message}]
            
            for msg in st.session_state.messages[-10:]:
                messages_for_api.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_api,
                max_completion_tokens=800,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            error_message = f"I encountered an error: {str(e)}. Please try again or check your API connection."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.rerun()

if not st.session_state.conversation_started:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #718096;">
        <p style="font-size: 18px; margin-bottom: 20px;">👋 Welcome! I'm your AI assistant.</p>
        <p style="font-size: 14px;">Ask me anything about blockchain analysis, security, or how to use this platform.</p>
    </div>
    """, unsafe_allow_html=True)
