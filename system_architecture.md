# QuantumGuard AI System Architecture

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        QUANTUM-SECURE BLOCKCHAIN ANALYZER                  │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION LAYER                            │
├───────────────┬───────────────────────────────────┬───────────────────────┤
│               │                                   │                       │
│  Blockchain   │           Banking Data            │   Etherscan Data      │
│  CSV Import   │           CSV Import              │   Converter           │
│               │                                   │                       │
└───────┬───────┴─────────────┬─────────────────────┴───────────┬───────────┘
        │                     │                                 │
        ▼                     ▼                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         POST-QUANTUM SECURITY LAYER                        │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│              Encryption/Decryption with Quantum-Resistant Algorithms      │
│                                                                           │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             PROCESSING LAYER                               │
├─────────────────┬─────────────────────────────────┬─────────────────────┬─┤
│                 │                                 │                     │ │
│  Data           │  Feature                        │  Network Graph      │ │
│  Preprocessing  │  Extraction                     │  Construction       │ │
│                 │                                 │                     │ │
└─────────────────┴─────────────────────────────────┴─────────────────────┴─┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              ANALYSIS LAYER                                │
├────────────────┬──────────────────┬───────────────────┬───────────────────┤
│                │                  │                   │                   │
│  Risk          │  Anomaly         │  Transaction      │  Pattern          │
│  Assessment    │  Detection       │  Network          │  Recognition      │
│                │                  │  Analysis         │                   │
└────────────────┴──────────────────┴───────────────────┴───────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            VISUALIZATION LAYER                             │
├───────────────┬───────────────┬───────────────────┬───────────────────────┤
│               │               │                   │                       │
│  Transaction  │  Risk         │  Anomaly          │  Transaction          │
│  Network      │  Heatmap      │  Visualization    │  Timeline            │
│               │               │                   │                       │
└───────────────┴───────────────┴───────────────────┴───────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            AI SEARCH LAYER                                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│                 OpenAI-Powered Natural Language Query Engine              │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           PERSISTENCE LAYER                                │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│                         PostgreSQL Database                               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Event-Driven Application Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ User Uploads│     │ Data Pre-   │     │ Analysis    │     │ Results     │
│ Transaction ├────►│ processing  ├────►│ Execution   ├────►│ Generation  │
│ Data        │     │ & Encryption│     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ AI Query    │     │ Context     │     │ OpenAI      │     │ Response    │
│ Submission  │◄────┤ Generation  │◄────┤ Processing  │◄────┤ Display     │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Component Descriptions

### Data Ingestion Layer
- **Blockchain CSV Import**: Handles uploading and initial processing of blockchain transaction files.
- **Banking Data CSV Import**: Processes bank transaction data files in CSV format.
- **Etherscan Data Converter**: Converts Etherscan export data into the standardized format.

### Post-Quantum Security Layer
- Implements quantum-resistant cryptographic algorithms to protect sensitive financial data.
- Ensures data remains secure against future quantum computing threats.

### Processing Layer
- **Data Preprocessing**: Cleans, normalizes, and formats the transaction data.
- **Feature Extraction**: Derives meaningful features from transaction data for machine learning models.
- **Network Graph Construction**: Builds transaction network graphs representing relationship between addresses.

### Analysis Layer
- **Risk Assessment**: Evaluates and scores transactions based on risk factors.
- **Anomaly Detection**: Identifies unusual transaction patterns using machine learning algorithms.
- **Transaction Network Analysis**: Analyzes network properties and identifies key entities.
- **Pattern Recognition**: Recognizes common transaction patterns and categorizes transactions.

### Visualization Layer
- **Transaction Network**: Interactive visualization of transaction relationships.
- **Risk Heatmap**: Visual representation of risk assessment results.
- **Anomaly Visualization**: Graphical display of detected anomalies.
- **Transaction Timeline**: Temporal visualization of transaction flow.

### AI Search Layer
- Provides natural language search functionality powered by OpenAI's GPT models.
- Allows users to query transaction data using conversational language.

### Persistence Layer
- PostgreSQL database for storing analysis sessions, transactions, and results.
- Enables saving, retrieving, and comparing multiple analysis sessions.

## Key Files and Their Functions

```
├── app.py                     # Main Streamlit application interface
├── blockchain_analyzer.py     # Core transaction analysis logic
├── data_processor.py          # Data preparation and feature extraction
├── ml_models.py               # Machine learning models for analysis
├── quantum_crypto.py          # Post-quantum encryption functionality
├── visualizations.py          # Network and data visualization components
├── database.py                # Database schema and operations
├── etherscan_converter.py     # Utility for converting Etherscan data
└── ai_search.py               # OpenAI-powered search functionality
```

## User Interaction Flow

1. User uploads transaction data (blockchain, bank, or Etherscan export)
2. System preprocesses data and applies quantum-secure encryption
3. Analysis runs (risk assessment, anomaly detection, network analysis)
4. Results displayed as interactive visualizations
5. User can query data using natural language via AI Search
6. Analysis session can be saved to database for future reference
7. Results can be exported in various formats (CSV, Excel, JSON)