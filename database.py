import os
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
from quantum_backend_security import encrypt_for_storage, decrypt_from_storage, get_backend_security_status

# Get the database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# Create SQLAlchemy engine and session with SSL configuration
if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "sslmode": "require",
            "connect_timeout": 10,
        },
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
        pool_timeout=30,     # Timeout after 30 seconds
        max_overflow=10      # Allow up to 10 overflow connections
    )
else:
    raise ValueError("DATABASE_URL environment variable is not set")
Session = sessionmaker(bind=engine)
Base = declarative_base()

class AnalysisSession(Base):
    """Model for storing analysis sessions"""
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    timestamp = Column(DateTime, default=datetime.now)
    dataset_name = Column(String(255))
    dataset_hash = Column(String(64))  # Store a hash of the dataset for identification
    risk_threshold = Column(Float)
    anomaly_sensitivity = Column(Float)
    description = Column(Text, nullable=True)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="session", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="session", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="session", cascade="all, delete-orphan")
    network_metrics = relationship("NetworkMetric", back_populates="session", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<AnalysisSession(id={self.id}, name='{self.name}', timestamp='{self.timestamp}')>"

class Transaction(Base):
    """Model for storing blockchain transactions"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'))
    from_address = Column(String(255))
    to_address = Column(String(255))
    value = Column(Float)
    timestamp = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=True)
    transaction_hash = Column(String(255), nullable=True)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="transactions")
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, from='{self.from_address}', to='{self.to_address}', value={self.value})>"

class RiskAssessment(Base):
    """Model for storing risk assessment results"""
    __tablename__ = 'risk_assessments'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'))
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    risk_score = Column(Float)
    risk_factors = Column(Text, nullable=True)
    risk_category = Column(String(50))  # Low, Medium, High
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="risk_assessments")
    
    def __repr__(self):
        return f"<RiskAssessment(id={self.id}, risk_score={self.risk_score}, category='{self.risk_category}')>"

class Anomaly(Base):
    """Model for storing anomaly detection results"""
    __tablename__ = 'anomalies'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'))
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="anomalies")
    
    def __repr__(self):
        return f"<Anomaly(id={self.id}, anomaly_score={self.anomaly_score}, is_anomaly={self.is_anomaly})>"

class NetworkMetric(Base):
    """Model for storing blockchain network metrics"""
    __tablename__ = 'network_metrics'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'))
    total_nodes = Column(Integer)
    total_edges = Column(Integer)
    avg_degree = Column(Float, nullable=True)
    clustering = Column(Float, nullable=True)
    connected_components = Column(Integer, nullable=True)
    largest_component_size = Column(Integer, nullable=True)
    top_addresses = Column(Text, nullable=True)  # Store as JSON string
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="network_metrics")
    
    def __repr__(self):
        return f"<NetworkMetric(id={self.id}, nodes={self.total_nodes}, edges={self.total_edges})>"

class AddressWatchlist(Base):
    """Model for storing address watchlists with custom labels"""
    __tablename__ = 'address_watchlists'
    
    id = Column(Integer, primary_key=True)
    address = Column(String(255), unique=True, nullable=False)
    label = Column(String(255), nullable=False)
    risk_level = Column(String(50), default='Medium')  # Low, Medium, High, Critical
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<AddressWatchlist(id={self.id}, address='{self.address}', label='{self.label}', risk='{self.risk_level}')>"

class SavedSearch(Base):
    """Model for storing saved search queries"""
    __tablename__ = 'saved_searches'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    search_query = Column(Text, nullable=False)
    search_type = Column(String(50), default='general')  # general, address, value, risk, etc.
    created_at = Column(DateTime, default=datetime.now)
    last_used = Column(DateTime, default=datetime.now)
    use_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<SavedSearch(id={self.id}, name='{self.name}', type='{self.search_type}')>"

def init_db():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(engine)

def save_analysis_to_db(
    session_name, 
    dataset_name,
    dataframe, 
    risk_assessment_df, 
    anomaly_indices, 
    network_metrics, 
    risk_threshold, 
    anomaly_sensitivity,
    description=None
):
    """
    Save analysis results to the database
    
    Args:
        session_name (str): Name for this analysis session
        dataset_name (str): Name of the dataset
        dataframe (pd.DataFrame): The original transaction dataframe
        risk_assessment_df (pd.DataFrame): Dataframe with risk assessment results
        anomaly_indices (list): List of indices representing anomalous transactions
        network_metrics (dict): Dictionary of network metrics
        risk_threshold (float): The risk threshold used
        anomaly_sensitivity (float): The anomaly sensitivity used
        description (str): Optional description of the analysis
        
    Returns:
        int: The ID of the created analysis session
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        session = Session()
        try:
            # Create quantum-safe dataset hash for identification
            import hashlib
            sample_data = dataframe.head(5).to_json()
            dataset_hash = hashlib.sha3_256(sample_data.encode()).hexdigest()
            
            # Encrypt sensitive description data
            encrypted_description = encrypt_for_storage(description or "", "database") if description else None
            
            # Create analysis session with quantum-safe encryption
            analysis_session = AnalysisSession(
                name=session_name,
                dataset_name=dataset_name,
                dataset_hash=dataset_hash,
                risk_threshold=risk_threshold,
                anomaly_sensitivity=anomaly_sensitivity,
                description=encrypted_description
            )
            session.add(analysis_session)
            session.flush()  # To get the ID without committing
            
            # Store transactions with quantum-safe encryption for sensitive addresses
            transactions = []
            for _, row in dataframe.iterrows():
                # Encrypt sensitive address data
                from_addr = encrypt_for_storage(row.get('from_address', 'unknown'), "database")
                to_addr = encrypt_for_storage(row.get('to_address', 'unknown'), "database")
                value = row.get('value', 0.0)
                
                timestamp = None
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    try:
                        timestamp = pd.to_datetime(row['timestamp'])
                    except:
                        pass
                        
                status = row.get('status', 'unknown')
                tx_hash = row.get('transaction_hash', None)
                
                transaction = Transaction(
                    session_id=analysis_session.id,
                    from_address=from_addr,
                    to_address=to_addr,
                    value=value,
                    timestamp=timestamp,
                    status=status,
                    transaction_hash=tx_hash
                )
                transactions.append(transaction)
            
            session.add_all(transactions)
            session.flush()
            
            # Store risk assessments
            if risk_assessment_df is not None:
                risk_assessments = []
                
                for i, row in risk_assessment_df.iterrows():
                    if i < len(transactions):  # Make sure we have a matching transaction
                        risk_assessment = RiskAssessment(
                            session_id=analysis_session.id,
                            transaction_id=transactions[i].id,
                            risk_score=row.get('risk_score', 0.0),
                            risk_factors=row.get('risk_factors', ''),
                            risk_category=row.get('risk_category', 'Low')
                        )
                        risk_assessments.append(risk_assessment)
                
                session.add_all(risk_assessments)
            
            # Store anomalies
            if anomaly_indices:
                anomalies = []
                
                for idx in anomaly_indices:
                    if idx < len(transactions):  # Make sure we have a matching transaction
                        anomaly = Anomaly(
                            session_id=analysis_session.id,
                            transaction_id=transactions[idx].id,
                            anomaly_score=1.0,  # We don't have actual scores in this implementation
                            is_anomaly=True
                        )
                        anomalies.append(anomaly)
                
                session.add_all(anomalies)
            
            # Store network metrics
            if network_metrics:
                network_metric = NetworkMetric(
                    session_id=analysis_session.id,
                    total_nodes=network_metrics.get('total_nodes', 0),
                    total_edges=network_metrics.get('total_edges', 0),
                    avg_degree=network_metrics.get('avg_degree', 0.0),
                    clustering=network_metrics.get('clustering', 0.0),
                    connected_components=network_metrics.get('connected_components', 0),
                    largest_component_size=network_metrics.get('largest_component_size', 0),
                    top_addresses=json.dumps(network_metrics.get('top_addresses', []))
                )
                session.add(network_metric)
            
            # Commit all changes
            session.commit()
            return analysis_session.id
            
        except Exception as e:
            session.rollback()
            retry_count += 1
            if retry_count >= max_retries:
                raise e
            # Wait before retrying
            import time
            time.sleep(1)
        finally:
            session.close()
    
    # If we get here, all retries failed
    raise Exception("Failed to save analysis after multiple attempts")

def get_analysis_sessions():
    """
    Fetch all analysis sessions from the database
    
    Returns:
        list: List of analysis session dictionaries
    """
    session = Session()
    try:
        sessions = session.query(AnalysisSession).order_by(AnalysisSession.timestamp.desc()).all()
        return [
            {
                'id': s.id,
                'name': s.name,
                'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_name': s.dataset_name,
                'risk_threshold': s.risk_threshold,
                'anomaly_sensitivity': s.anomaly_sensitivity,
                'description': s.description
            }
            for s in sessions
        ]
    finally:
        session.close()

def get_analysis_by_id(session_id):
    """
    Fetch a specific analysis session with all its related data
    
    Args:
        session_id (int): The ID of the analysis session
        
    Returns:
        dict: Dictionary with all analysis data
    """
    session = Session()
    try:
        analysis = session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        
        if not analysis:
            return None
        
        # Get all transactions for this session
        transactions = session.query(Transaction).filter(Transaction.session_id == session_id).all()
        tx_data = [
            {
                'id': tx.id,
                'from_address': tx.from_address,
                'to_address': tx.to_address,
                'value': tx.value,
                'timestamp': tx.timestamp.strftime('%Y-%m-%d %H:%M:%S') if tx.timestamp else None,
                'status': tx.status,
                'transaction_hash': tx.transaction_hash
            }
            for tx in transactions
        ]
        
        # Get risk assessments
        risks = session.query(RiskAssessment).filter(RiskAssessment.session_id == session_id).all()
        risk_data = [
            {
                'id': r.id,
                'transaction_id': r.transaction_id,
                'risk_score': r.risk_score,
                'risk_factors': r.risk_factors,
                'risk_category': r.risk_category
            }
            for r in risks
        ]
        
        # Get anomalies
        anomalies = session.query(Anomaly).filter(Anomaly.session_id == session_id).all()
        anomaly_data = [
            {
                'id': a.id,
                'transaction_id': a.transaction_id,
                'anomaly_score': a.anomaly_score,
                'is_anomaly': a.is_anomaly
            }
            for a in anomalies
        ]
        
        # Get network metrics
        network_metrics = session.query(NetworkMetric).filter(NetworkMetric.session_id == session_id).first()
        if network_metrics:
            network_data = {
                'total_nodes': network_metrics.total_nodes,
                'total_edges': network_metrics.total_edges,
                'avg_degree': network_metrics.avg_degree,
                'clustering': network_metrics.clustering,
                'connected_components': network_metrics.connected_components,
                'largest_component_size': network_metrics.largest_component_size,
                'top_addresses': json.loads(network_metrics.top_addresses) if network_metrics.top_addresses and isinstance(network_metrics.top_addresses, str) else []
            }
        else:
            network_data = {}
        
        # Build the response
        return {
            'id': analysis.id,
            'name': analysis.name,
            'timestamp': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_name': analysis.dataset_name,
            'risk_threshold': analysis.risk_threshold,
            'anomaly_sensitivity': analysis.anomaly_sensitivity,
            'description': analysis.description,
            'transactions': tx_data,
            'risk_assessments': risk_data,
            'anomalies': anomaly_data,
            'network_metrics': network_data
        }
    finally:
        session.close()

def delete_analysis_session(session_id):
    """
    Delete an analysis session and all related data
    
    Args:
        session_id (int): The ID of the analysis session to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    session = Session()
    try:
        analysis = session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        
        if not analysis:
            return False
        
        session.delete(analysis)
        session.commit()
        return True
    except:
        session.rollback()
        return False
    finally:
        session.close()

# Address Watchlist Management Functions
def add_address_to_watchlist(address: str, label: str, risk_level: str = 'Medium', notes: str = ''):
    """Add an address to the watchlist"""
    session = Session()
    try:
        # Check if address already exists
        existing = session.query(AddressWatchlist).filter_by(address=address).first()
        if existing:
            # Update existing entry
            existing.label = label
            existing.risk_level = risk_level
            existing.notes = notes
            existing.updated_at = datetime.now()
            existing.is_active = True
            watchlist_entry = existing
        else:
            # Create new entry
            watchlist_entry = AddressWatchlist(
                address=address,
                label=label,
                risk_level=risk_level,
                notes=notes
            )
            session.add(watchlist_entry)
        
        session.commit()
        return watchlist_entry.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_watchlist_addresses():
    """Get all active watchlist addresses"""
    session = Session()
    try:
        watchlist = session.query(AddressWatchlist).filter_by(is_active=True).all()
        return [{
            'id': entry.id,
            'address': entry.address,
            'label': entry.label,
            'risk_level': entry.risk_level,
            'notes': entry.notes,
            'created_at': entry.created_at
        } for entry in watchlist]
    except Exception as e:
        raise e
    finally:
        session.close()

def remove_address_from_watchlist(address_id: int):
    """Remove an address from the watchlist"""
    session = Session()
    try:
        watchlist_entry = session.query(AddressWatchlist).filter_by(id=address_id).first()
        if watchlist_entry:
            watchlist_entry.is_active = False
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def check_addresses_against_watchlist(addresses: list):
    """Check if any addresses are in the watchlist"""
    session = Session()
    try:
        watchlist_addresses = session.query(AddressWatchlist).filter(
            AddressWatchlist.address.in_(addresses),
            AddressWatchlist.is_active == True
        ).all()
        
        return [{
            'address': entry.address,
            'label': entry.label,
            'risk_level': entry.risk_level,
            'notes': entry.notes
        } for entry in watchlist_addresses]
    except Exception as e:
        raise e
    finally:
        session.close()

# Saved Search Management Functions
def save_search_query(name: str, query: str, search_type: str = 'general'):
    """Save a search query for future use"""
    session = Session()
    try:
        saved_search = SavedSearch(
            name=name,
            search_query=query,
            search_type=search_type
        )
        session.add(saved_search)
        session.commit()
        return saved_search.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_saved_searches():
    """Get all active saved searches"""
    session = Session()
    try:
        searches = session.query(SavedSearch).filter_by(is_active=True).order_by(SavedSearch.last_used.desc()).all()
        return [{
            'id': search.id,
            'name': search.name,
            'query': search.search_query,
            'type': search.search_type,
            'last_used': search.last_used,
            'use_count': search.use_count
        } for search in searches]
    except Exception as e:
        raise e
    finally:
        session.close()

def use_saved_search(search_id: int):
    """Update usage statistics for a saved search"""
    session = Session()
    try:
        saved_search = session.query(SavedSearch).filter_by(id=search_id).first()
        if saved_search:
            saved_search.last_used = datetime.now()
            saved_search.use_count += 1
            session.commit()
            return {
                'id': saved_search.id,
                'name': saved_search.name,
                'query': saved_search.search_query,
                'type': saved_search.search_type
            }
        return None
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def delete_saved_search(search_id: int):
    """Delete a saved search"""
    session = Session()
    try:
        saved_search = session.query(SavedSearch).filter_by(id=search_id).first()
        if saved_search:
            saved_search.is_active = False
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Initialize the database (creates tables if they don't exist)
init_db()