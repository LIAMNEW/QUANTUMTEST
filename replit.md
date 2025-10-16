# Overview

QuantumGuard AI is a comprehensive blockchain transaction analysis platform designed to analyze transactions, detect anomalies, and assess risks using advanced machine learning, AI-powered analytics, and quantum-resistant cryptography. It provides intelligent insights through a Streamlit-based web interface, emphasizing post-quantum security standards and enterprise-grade security for financial data protection. The platform also includes real-world fraud pattern detection for bank and credit card transactions and an intelligent AI agent for app control and data analysis.

# User Preferences

Preferred communication style: Simple, everyday language.
Security Focus: Backend quantum safety for customer financial data protection (not transaction analysis testing).

# System Architecture

## Enterprise Security Infrastructure
- **Production Quantum Security**: AES-256-GCM encryption with PBKDF2 key derivation (480,000 iterations), RSA-4096 hybrid encryption, and certified cryptographic libraries.
- **Enterprise Key Management**: Secure key vault with master key encryption, automatic rotation, and audit trails.
- **Multi-Factor Authentication**: TOTP-based MFA with backup codes, rate limiting, and account lockout protection.
- **API Security Middleware**: Rate limiting, DDoS protection, IP blocking, and suspicious pattern detection.
- **Backup & Disaster Recovery**: Automated encrypted backups, compression, retention policies, and disaster recovery procedures.

## Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard for blockchain analysis with a modern glassmorphism design system.
- **UI/UX**: Features frosted glass effects, gradient backgrounds, smooth animations (slide-in-up, fade-in, hover), skeleton screens, animated progress bars, modern alerts, and enhanced metric cards.
- **Accessibility**: ARIA labels, focus indicators, reduced motion support, and high contrast mode.
- **Interactive Visualizations**: Plotly-based visualizations for network graphs, risk heatmaps, anomaly detection, and transaction timelines.
- **Export Functionality**: Multi-format export (CSV, JSON, Excel) with one-click downloads.

## Backend Architecture
- **Modular Processing Pipeline**: Separate modules for data processing, blockchain analysis, and machine learning models.
- **AI Analytics Engine**: Advanced AI analytics combining clustering, behavioral patterns, risk correlation, and temporal analysis.
- **Quantum-Resistant Security**: Post-quantum cryptography implementation for secure data encryption and decryption.
- **Data Converters**: Specialized converters for various data sources, including Etherscan API data.
- **AI Agent System**: Dual-assistant system with an Advanced AI Agent (GPT-4o) for app control and user assistance, and a Data AI Assistant (GPT-4o) for context-aware data analysis.

## Data Storage Solutions
- **PostgreSQL Database**: SQLAlchemy-based database layer for analysis sessions, transactions, risk assessments, anomalies, and network metrics.
- **Session Persistence**: Complete analysis sessions can be saved, retrieved, and managed.
- **SSL-Secured Connections**: Database connections use SSL with connection pooling and timeout management.

## Machine Learning & AI Components
- **Advanced Anomaly Detection**: Ensemble methods including LSTM Autoencoders, Variational Autoencoders, and Graph Neural Networks for sophisticated pattern recognition and transaction network analysis.
- **Risk Assessment**: Multi-factor risk scoring using transaction patterns, network metrics, and behavioral analysis, including real-world fraud pattern detection (time-based, merchant analysis, geographic risk, structuring, velocity anomalies).
- **Online Learning**: Continuous model updates with concept drift detection and user feedback loops.
- **AI Integration**: OpenAI's GPT-5 for enhanced reasoning, natural language queries, and advanced insights generation.

# External Dependencies

## Third-Party APIs
- **OpenAI API**: Used for AI-powered search, natural language processing, and generating intelligent insights.
- **Etherscan API**: For importing and converting blockchain transaction data.

## Machine Learning Libraries
- **scikit-learn**: Core machine learning functionalities including RandomForestRegressor, IsolationForest, LinearRegression, StandardScaler, LabelEncoder, DBSCAN.
- **pandas & numpy**: Data manipulation, analysis, and numerical computing.
- **NetworkX**: Graph analysis and network visualization for transaction networks.

## Visualization and UI
- **Streamlit**: Web application framework for the interactive dashboard.
- **Plotly**: Interactive plotting library for dynamic visualizations.

## Database and Storage
- **SQLAlchemy**: Object-relational mapping for database interactions.
- **PostgreSQL**: Primary database for storing analysis results, user sessions, and transaction data.

## Security and Cryptography
- **Custom Post-Quantum Cryptography**: Simplified Kyber-like KEM implementation.
- **Standard Cryptographic Libraries**: hashlib, base64, os.urandom.

## Data Processing
- **datetime**: Temporal analysis and timestamp processing.
- **json**: Data serialization and API communication.
- **argparse**: Command-line interface for data conversion tools.