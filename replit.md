# Overview

QuantumGuard AI is a comprehensive blockchain transaction analysis platform that combines advanced machine learning, quantum-resistant cryptography, and AI-powered analytics. The system is designed to analyze blockchain transactions, detect anomalies, assess risks, and provide intelligent insights while maintaining post-quantum security standards. It features a Streamlit-based web interface for interactive analysis and visualization of blockchain data.

# User Preferences

Preferred communication style: Simple, everyday language.
Security Focus: Backend quantum safety for customer financial data protection (not transaction analysis testing).

# System Architecture

## Enterprise Security Infrastructure
- **Production Quantum Security**: AES-256-GCM encryption with PBKDF2 key derivation, RSA-4096 hybrid encryption, and certified cryptographic libraries
- **Enterprise Key Management**: Secure key vault with master key encryption, automatic key rotation, and audit trails
- **Multi-Factor Authentication**: TOTP-based MFA with backup codes, rate limiting, and account lockout protection
- **API Security Middleware**: Rate limiting, DDoS protection, IP blocking, and suspicious pattern detection
- **Backup & Disaster Recovery**: Automated encrypted backups, compression, retention policies, and disaster recovery procedures

## Frontend Architecture
- **Streamlit Web Application**: The main user interface built with Streamlit (`app.py`) providing an interactive dashboard for blockchain analysis
- **Interactive Visualizations**: Plotly-based visualizations (`visualizations.py`) for network graphs, risk heatmaps, anomaly detection plots, and transaction timelines
- **Session Management**: Built-in session state management for maintaining user data and analysis results across interactions

## Backend Architecture
- **Modular Processing Pipeline**: Separate modules for data processing (`data_processor.py`), blockchain analysis (`blockchain_analyzer.py`), and machine learning models (`ml_models.py`)
- **AI Analytics Engine**: Advanced AI analytics module (`advanced_ai_analytics.py`) providing multimodal analysis combining clustering, behavioral patterns, risk correlation, and temporal analysis
- **Quantum-Resistant Security**: Post-quantum cryptography implementation (`quantum_crypto.py`) for secure data encryption and decryption
- **Data Converters**: Specialized converters for different data sources, including Etherscan API data (`etherscan_converter.py`)

## Data Storage Solutions
- **PostgreSQL Database**: SQLAlchemy-based database layer (`database.py`) with models for analysis sessions, transactions, risk assessments, anomalies, and network metrics
- **Session Persistence**: Complete analysis sessions can be saved, retrieved, and managed through the database layer
- **SSL-Secured Connections**: Database connections use SSL with connection pooling and timeout management

## Machine Learning Components
- **Anomaly Detection**: Isolation Forest-based anomaly detection with configurable sensitivity
- **Risk Assessment**: Multi-factor risk scoring using transaction patterns, network metrics, and behavioral analysis
- **Clustering Analysis**: DBSCAN clustering for transaction pattern identification
- **Network Analysis**: NetworkX-based graph analysis for transaction network insights

## AI Integration
- **OpenAI Integration**: AI-powered search and insights generation (`ai_search.py`) using OpenAI's API for natural language queries and analysis
- **Advanced Analytics**: Comprehensive multimodal analysis combining multiple AI approaches for deep transaction insights
- **Intelligent Pattern Recognition**: AI-driven identification of suspicious patterns and behavioral anomalies

# External Dependencies

## Third-Party APIs
- **OpenAI API**: Used for AI-powered search, natural language processing, and generating intelligent insights from transaction data
- **Etherscan API Integration**: Support for importing and converting Etherscan transaction data

## Quantum Security Testing
- **Comprehensive Cryptographic Validation**: Built-in quantum security test suite (`quantum_security_test.py`) validates post-quantum cryptographic implementation
- **Security Metrics**: Tests key generation entropy, encryption security, quantum attack resistance, and performance benchmarks
- **Standards Compliance**: Validates compliance with NIST post-quantum cryptography standards and resistance to Shor's and Grover's quantum algorithms

## Machine Learning Libraries
- **scikit-learn**: Core machine learning functionality including RandomForestRegressor, IsolationForest, LinearRegression, StandardScaler, LabelEncoder, DBSCAN, and various metrics
- **pandas & numpy**: Data manipulation, analysis, and numerical computing
- **NetworkX**: Graph analysis and network visualization for transaction networks

## Visualization and UI
- **Streamlit**: Web application framework for the interactive dashboard
- **Plotly**: Interactive plotting library for creating dynamic visualizations including network graphs, heatmaps, and timelines

## Database and Storage
- **SQLAlchemy**: Object-relational mapping for PostgreSQL database interactions
- **PostgreSQL**: Primary database for storing analysis results, user sessions, and transaction data

## Security and Cryptography
- **Post-Quantum Cryptography**: Custom implementation of quantum-resistant encryption algorithms (simplified Kyber-like KEM)
- **Standard Cryptographic Libraries**: hashlib, base64, and os.urandom for security operations

## Data Processing
- **datetime**: Temporal analysis and timestamp processing
- **json**: Data serialization and API communication
- **argparse**: Command-line interface for data conversion tools