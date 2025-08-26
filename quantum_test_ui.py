#!/usr/bin/env python3
"""
QuantumGuard AI - Interactive Quantum Security Test Interface

Streamlit interface for running and displaying quantum cryptography security tests.
"""

import streamlit as st
import time
import json
from quantum_security_test import run_quantum_security_test, QuantumSecurityTester
from quantum_crypto import SECURITY_LEVEL, MODULUS, NOISE_PARAMETER

def display_quantum_security_dashboard():
    """Display the quantum security testing dashboard"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2>🛡️ Quantum Security Verification</h2>
        <p>Comprehensive testing suite to verify the quantum-resistant properties of QuantumGuard AI's cryptographic implementation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Security parameters overview
    st.markdown("### Current Security Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.metric("Security Level", f"{SECURITY_LEVEL} bits", help="Cryptographic security level in bits")
    
    with param_col2:
        st.metric("Algorithm Type", "Lattice-Based", help="Post-quantum cryptographic approach")
    
    with param_col3:
        st.metric("Quantum Resistant", "Yes", help="Resistant to quantum computer attacks")
    
    # Test categories
    st.markdown("### Available Security Tests")
    
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        st.markdown("""
        **🔑 Key Generation Tests:**
        - Entropy analysis
        - Randomness verification
        - Uniqueness validation
        
        **🔒 Encryption Security Tests:**
        - Semantic security
        - Ciphertext randomness
        - Correctness verification
        """)
    
    with test_col2:
        st.markdown("""
        **⚛️ Quantum Attack Resistance:**
        - Shor's algorithm resistance
        - Grover's algorithm resistance
        - LWE problem hardness
        
        **⚡ Performance Benchmarks:**
        - Throughput measurements
        - Scalability analysis
        - Efficiency validation
        """)
    
    # Test execution
    st.markdown("### Run Security Tests")
    
    test_type = st.selectbox(
        "Select test type to run:",
        [
            "Complete Security Suite",
            "Key Generation Only", 
            "Encryption Security Only",
            "Quantum Resistance Only",
            "Performance Benchmarks Only"
        ]
    )
    
    if st.button("🚀 Run Security Tests", type="primary"):
        run_security_tests(test_type)

def run_security_tests(test_type: str):
    """Execute the selected security tests"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        tester = QuantumSecurityTester()
        
        if test_type == "Complete Security Suite":
            status_text.text("Running comprehensive security test suite...")
            progress_bar.progress(10)
            
            results = tester.run_comprehensive_test_suite()
            progress_bar.progress(100)
            
            display_comprehensive_results(results)
            
        elif test_type == "Key Generation Only":
            status_text.text("Testing key generation entropy...")
            progress_bar.progress(50)
            
            results = tester.test_key_generation_entropy()
            progress_bar.progress(100)
            
            display_key_generation_results(results)
            
        elif test_type == "Encryption Security Only":
            status_text.text("Testing encryption security...")
            progress_bar.progress(50)
            
            results = tester.test_encryption_security()
            progress_bar.progress(100)
            
            display_encryption_results(results)
            
        elif test_type == "Quantum Resistance Only":
            status_text.text("Testing quantum attack resistance...")
            progress_bar.progress(50)
            
            results = tester.test_quantum_attack_resistance()
            progress_bar.progress(100)
            
            display_quantum_resistance_results(results)
            
        elif test_type == "Performance Benchmarks Only":
            status_text.text("Running performance benchmarks...")
            progress_bar.progress(50)
            
            results = tester.test_performance_benchmarks()
            progress_bar.progress(100)
            
            display_performance_results(results)
        
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Test execution failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_comprehensive_results(results: dict):
    """Display comprehensive test results"""
    
    st.markdown("## 🔒 Comprehensive Security Test Results")
    
    # Overall status
    status = results['overall_security_status']
    if status == "QUANTUM-SAFE":
        st.success(f"✅ **Overall Status: {status}**")
        st.balloons()
    else:
        st.error(f"❌ **Overall Status: {status}**")
    
    # Summary metrics
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Security Level", f"{results['security_level']} bits")
    
    with summary_col2:
        st.metric("Tests Passed", f"{results['tests_passed']}/{results['total_tests']}")
    
    with summary_col3:
        pass_rate = (results['tests_passed'] / results['total_tests']) * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    with summary_col4:
        st.metric("Test Timestamp", results['test_timestamp'])
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔑 Key Generation", 
        "🔒 Encryption Security", 
        "⚛️ Quantum Resistance", 
        "⚡ Performance"
    ])
    
    with tab1:
        display_key_generation_results(results['detailed_results']['key_generation'])
    
    with tab2:
        display_encryption_results(results['detailed_results']['encryption_security'])
    
    with tab3:
        display_quantum_resistance_results(results['detailed_results']['quantum_resistance'])
    
    with tab4:
        display_performance_results(results['detailed_results']['performance'])
    
    # Recommendations
    st.markdown("## 📋 Security Recommendations")
    for rec in results['recommendations']:
        if "✅" in rec:
            st.success(rec)
        elif "⚠️" in rec:
            st.warning(rec)
        else:
            st.info(rec)

def display_key_generation_results(results: dict):
    """Display key generation test results"""
    
    st.markdown("### Key Generation Security Analysis")
    
    if results['status'] == "PASS":
        st.success("✅ Key generation security: PASSED")
    else:
        st.error("❌ Key generation security: FAILED")
    
    # Metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Keys Generated", results['total_keys_generated'])
        st.metric("Uniqueness Rate", f"{results['uniqueness_rate']:.1f}%")
    
    with metric_col2:
        st.metric("Average Entropy", f"{results['average_entropy']:.2f} bits")
        entropy_status = "High" if results['entropy_threshold_passed'] else "Low"
        st.metric("Entropy Quality", entropy_status)
    
    with metric_col3:
        st.metric("Generation Speed", f"{results['generation_time_per_key']*1000:.2f} ms/key")
    
    # Analysis
    st.markdown("### Analysis")
    if results['uniqueness_rate'] == 100:
        st.success("🎯 All generated keys are unique - excellent randomness")
    else:
        st.warning(f"⚠️ {100-results['uniqueness_rate']:.1f}% key collision rate detected")
    
    if results['entropy_threshold_passed']:
        st.success("🔒 High entropy achieved - cryptographically secure randomness")
    else:
        st.warning("⚠️ Low entropy detected - may indicate weak randomness source")

def display_encryption_results(results: dict):
    """Display encryption security test results"""
    
    st.markdown("### Encryption Security Analysis")
    
    if results['status'] == "PASS":
        st.success("✅ Encryption security: PASSED")
    else:
        st.error("❌ Encryption security: FAILED")
    
    # Performance metrics
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Avg Encryption Time", f"{results['average_encryption_time']*1000:.2f} ms")
    
    with perf_col2:
        st.metric("Avg Decryption Time", f"{results['average_decryption_time']*1000:.2f} ms")
    
    with perf_col3:
        st.metric("Ciphertext Entropy", f"{results['average_ciphertext_entropy']:.2f} bits")
    
    # Randomness test results
    st.markdown("### Randomness Analysis")
    randomness = results['randomness_test']
    
    if randomness['passed']:
        st.success(f"✅ Semantic security verified: {randomness['unique_ciphertexts']}/{randomness['total_tests']} unique ciphertexts")
    else:
        st.error(f"❌ Semantic security failed: Only {randomness['unique_ciphertexts']}/{randomness['total_tests']} unique ciphertexts")
    
    if results['high_entropy_achieved']:
        st.success("🔒 High ciphertext entropy achieved - strong randomness")
    else:
        st.warning("⚠️ Low ciphertext entropy - potential security concern")

def display_quantum_resistance_results(results: dict):
    """Display quantum attack resistance results"""
    
    st.markdown("### Quantum Attack Resistance Analysis")
    
    overall = results['overall_quantum_resistance']
    if overall['secure']:
        st.success(f"✅ **Status: {overall['status']}**")
    else:
        st.error(f"❌ **Status: {overall['status']}**")
    
    # Individual test results
    resistance_tests = [
        ("Lattice Security", results['lattice_based_security']),
        ("LWE Hardness", results['lwe_problem_hardness']),
        ("Key Size Analysis", results['key_size_analysis']),
        ("Shor Resistance", results['shor_resistance']),
        ("Grover Resistance", results['grover_resistance'])
    ]
    
    for test_name, test_result in resistance_tests:
        with st.expander(f"📊 {test_name}"):
            if test_result.get('secure', False):
                st.success(f"✅ {test_name}: SECURE")
            else:
                st.error(f"❌ {test_name}: VULNERABLE")
            
            # Display test details
            for key, value in test_result.items():
                if key != 'secure':
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def display_performance_results(results: dict):
    """Display performance benchmark results"""
    
    st.markdown("### Performance Benchmark Results")
    
    # Overall metrics
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Key Generation", f"{results['key_generation_time']*1000:.2f} ms")
    
    with perf_col2:
        st.metric("Overall Throughput", results['overall_throughput'])
    
    with perf_col3:
        st.metric("Performance Grade", results['performance_grade'])
    
    # Performance by data size
    st.markdown("### Performance by Data Size")
    
    performance_data = results['performance_by_data_size']
    
    for size, metrics in performance_data.items():
        with st.expander(f"📈 {size.replace('_', ' ').title()}"):
            size_col1, size_col2, size_col3 = st.columns(3)
            
            with size_col1:
                st.metric("Encryption Time", f"{metrics['avg_encryption_time']*1000:.2f} ms")
            
            with size_col2:
                st.metric("Decryption Time", f"{metrics['avg_decryption_time']*1000:.2f} ms")
            
            with size_col3:
                st.metric("Throughput", f"{metrics['throughput_mbps']:.2f} MB/s")

# Main execution
if __name__ == "__main__":
    st.set_page_config(
        page_title="QuantumGuard AI - Security Testing",
        page_icon="🛡️",
        layout="wide"
    )
    
    display_quantum_security_dashboard()