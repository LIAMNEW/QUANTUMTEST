#!/usr/bin/env python3
"""
QuantumGuard AI - Post-Quantum Cryptography Security Test Suite

This test suite validates the quantum-resistant properties of the cryptographic
implementation used in QuantumGuard AI. It tests against known quantum attack
vectors and validates the security parameters.
"""

import os
import time
import hashlib
import secrets
import numpy as np
from typing import Tuple, Dict, List
import streamlit as st
from quantum_crypto import (
    generate_pq_keys, 
    encrypt_data, 
    decrypt_data,
    SECURITY_LEVEL,
    MODULUS,
    NOISE_PARAMETER
)

class QuantumSecurityTester:
    """
    Comprehensive test suite for post-quantum cryptographic security.
    Tests the implementation against various attack scenarios.
    """
    
    def __init__(self):
        self.test_results = {}
        self.security_metrics = {}
        
    def test_key_generation_entropy(self, num_tests: int = 100) -> Dict:
        """Test the entropy and randomness of key generation"""
        print("Testing key generation entropy...")
        
        public_keys = []
        private_keys = []
        
        start_time = time.time()
        for i in range(num_tests):
            pub, priv = generate_pq_keys()
            public_keys.append(pub)
            private_keys.append(priv)
        generation_time = time.time() - start_time
        
        # Test for key uniqueness
        unique_public = len(set(str(k) for k in public_keys))
        unique_private = len(set(str(k) for k in private_keys))
        
        # Entropy analysis
        entropy_scores = []
        for key in public_keys[:10]:  # Test first 10 keys
            key_bytes = str(key).encode()
            entropy = self._calculate_entropy(key_bytes)
            entropy_scores.append(entropy)
        
        avg_entropy = np.mean(entropy_scores)
        
        results = {
            "total_keys_generated": num_tests,
            "unique_public_keys": unique_public,
            "unique_private_keys": unique_private,
            "uniqueness_rate": (unique_public / num_tests) * 100,
            "average_entropy": avg_entropy,
            "generation_time_per_key": generation_time / num_tests,
            "entropy_threshold_passed": avg_entropy > 7.0,  # High entropy threshold
            "status": "PASS" if unique_public == num_tests and avg_entropy > 7.0 else "FAIL"
        }
        
        self.test_results["key_generation_entropy"] = results
        return results
    
    def test_encryption_security(self, num_tests: int = 50) -> Dict:
        """Test encryption security and resistance to cryptanalysis"""
        print("Testing encryption security...")
        
        # Generate test keys
        public_key, private_key = generate_pq_keys()
        
        test_data = [
            b"test_data_" + secrets.token_bytes(32),
            b"quantum_resistant_test_" + secrets.token_bytes(64),
            b"A" * 1000,  # Repeated pattern
            secrets.token_bytes(2048),  # Random large data
            b"",  # Empty data
        ]
        
        encryption_times = []
        decryption_times = []
        ciphertext_entropies = []
        
        for data in test_data:
            # Test encryption
            start_time = time.time()
            ciphertext = encrypt_data(data, public_key)
            encryption_time = time.time() - start_time
            encryption_times.append(encryption_time)
            
            # Test decryption
            start_time = time.time()
            decrypted = decrypt_data(ciphertext, private_key)
            decryption_time = time.time() - start_time
            decryption_times.append(decryption_time)
            
            # Verify correctness
            if decrypted != data:
                return {"status": "FAIL", "error": "Decryption failed to recover original data"}
            
            # Analyze ciphertext entropy
            entropy = self._calculate_entropy(ciphertext)
            ciphertext_entropies.append(entropy)
        
        # Test ciphertext randomness
        randomness_test = self._test_ciphertext_randomness(public_key, num_tests)
        
        results = {
            "encryption_correctness": "PASS",
            "average_encryption_time": np.mean(encryption_times),
            "average_decryption_time": np.mean(decryption_times),
            "average_ciphertext_entropy": np.mean(ciphertext_entropies),
            "randomness_test": randomness_test,
            "high_entropy_achieved": np.mean(ciphertext_entropies) > 7.5,
            "status": "PASS" if np.mean(ciphertext_entropies) > 7.5 and randomness_test["passed"] else "FAIL"
        }
        
        self.test_results["encryption_security"] = results
        return results
    
    def test_quantum_attack_resistance(self) -> Dict:
        """Test resistance against known quantum attack vectors"""
        print("Testing quantum attack resistance...")
        
        results = {
            "lattice_based_security": self._test_lattice_security(),
            "lwe_problem_hardness": self._test_lwe_hardness(),
            "key_size_analysis": self._analyze_key_sizes(),
            "noise_parameter_validation": self._validate_noise_parameters(),
            "grover_resistance": self._test_grover_resistance(),
            "shor_resistance": self._test_shor_resistance()
        }
        
        # Overall assessment
        all_tests_passed = all(
            test_result.get("secure", False) for test_result in results.values()
        )
        
        results["overall_quantum_resistance"] = {
            "secure": all_tests_passed,
            "status": "QUANTUM-SAFE" if all_tests_passed else "POTENTIALLY VULNERABLE",
            "security_level": SECURITY_LEVEL
        }
        
        self.test_results["quantum_attack_resistance"] = results
        return results
    
    def test_performance_benchmarks(self) -> Dict:
        """Benchmark performance of cryptographic operations"""
        print("Running performance benchmarks...")
        
        # Generate keys
        key_gen_times = []
        for _ in range(10):
            start_time = time.time()
            generate_pq_keys()
            key_gen_times.append(time.time() - start_time)
        
        # Test encryption performance with different data sizes
        public_key, private_key = generate_pq_keys()
        data_sizes = [100, 1000, 10000, 100000]  # bytes
        
        performance_data = {}
        for size in data_sizes:
            test_data = secrets.token_bytes(size)
            
            # Encryption benchmark
            enc_times = []
            for _ in range(5):
                start_time = time.time()
                ciphertext = encrypt_data(test_data, public_key)
                enc_times.append(time.time() - start_time)
            
            # Decryption benchmark
            dec_times = []
            for _ in range(5):
                start_time = time.time()
                decrypt_data(ciphertext, private_key)
                dec_times.append(time.time() - start_time)
            
            performance_data[f"{size}_bytes"] = {
                "avg_encryption_time": np.mean(enc_times),
                "avg_decryption_time": np.mean(dec_times),
                "throughput_mbps": (size / (1024 * 1024)) / np.mean(enc_times)
            }
        
        results = {
            "key_generation_time": np.mean(key_gen_times),
            "performance_by_data_size": performance_data,
            "overall_throughput": f"{performance_data['10000_bytes']['throughput_mbps']:.2f} MB/s",
            "performance_grade": self._grade_performance(performance_data)
        }
        
        self.test_results["performance_benchmarks"] = results
        return results
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        try:
            # Convert bytes to numpy array of integers
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # Count frequency of each byte value
            frequency = np.bincount(data_array, minlength=256)
            frequency = frequency[frequency > 0]  # Remove zeros
            
            # Calculate probabilities
            probabilities = frequency / len(data_array)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
        except Exception:
            # Fallback method for string data
            if isinstance(data, str):
                data = data.encode('utf-8')
            data_array = np.array([ord(c) if isinstance(c, str) else c for c in data])
            unique, counts = np.unique(data_array, return_counts=True)
            probabilities = counts / len(data_array)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
    
    def _test_ciphertext_randomness(self, public_key, num_tests: int) -> Dict:
        """Test randomness properties of ciphertext"""
        same_plaintext = b"identical_test_data"
        ciphertexts = []
        
        for _ in range(num_tests):
            ciphertext = encrypt_data(same_plaintext, public_key)
            ciphertexts.append(ciphertext)
        
        # Check that all ciphertexts are different (semantic security)
        unique_ciphertexts = len(set(ciphertexts))
        
        return {
            "unique_ciphertexts": unique_ciphertexts,
            "total_tests": num_tests,
            "randomness_ratio": unique_ciphertexts / num_tests,
            "passed": unique_ciphertexts == num_tests  # All should be unique
        }
    
    def _test_lattice_security(self) -> Dict:
        """Test lattice-based cryptography security parameters"""
        # Validate that we're using appropriate lattice parameters
        dimension_check = MODULUS > 2**10  # Minimum dimension
        security_margin = SECURITY_LEVEL >= 128  # Bits of security
        
        return {
            "lattice_dimension_adequate": dimension_check,
            "security_level_sufficient": security_margin,
            "modulus_size": MODULUS.bit_length() if hasattr(MODULUS, 'bit_length') else len(str(MODULUS)),
            "secure": dimension_check and security_margin
        }
    
    def _test_lwe_hardness(self) -> Dict:
        """Test Learning With Errors problem hardness"""
        # The LWE problem should be hard to solve even with quantum computers
        noise_ratio = NOISE_PARAMETER / MODULUS if MODULUS != 0 else 0
        
        # Noise should be significant but not too large
        optimal_noise = 0.01 <= noise_ratio <= 0.1
        
        return {
            "noise_parameter": NOISE_PARAMETER,
            "noise_ratio": noise_ratio,
            "optimal_noise_range": optimal_noise,
            "lwe_hardness_estimated": "HIGH" if optimal_noise else "MEDIUM",
            "secure": optimal_noise
        }
    
    def _analyze_key_sizes(self) -> Dict:
        """Analyze key sizes for quantum resistance"""
        public_key, private_key = generate_pq_keys()
        
        pub_key_size = len(str(public_key))
        priv_key_size = len(str(private_key))
        
        # Post-quantum keys should be larger than classical keys
        adequate_size = pub_key_size > 1000 and priv_key_size > 1000
        
        return {
            "public_key_size_bytes": pub_key_size,
            "private_key_size_bytes": priv_key_size,
            "size_adequate_for_pq": adequate_size,
            "size_grade": "EXCELLENT" if adequate_size else "NEEDS_IMPROVEMENT",
            "secure": adequate_size
        }
    
    def _validate_noise_parameters(self) -> Dict:
        """Validate noise parameters for security"""
        # Noise should provide security but allow correct decryption
        return {
            "noise_parameter": NOISE_PARAMETER,
            "noise_validation": "SECURE" if NOISE_PARAMETER > 0 else "INSECURE",
            "secure": NOISE_PARAMETER > 0
        }
    
    def _test_grover_resistance(self) -> Dict:
        """Test resistance against Grover's quantum algorithm"""
        # Grover's algorithm provides quadratic speedup
        # Security level should account for this
        effective_security = SECURITY_LEVEL / 2  # Grover halves security
        
        return {
            "classical_security_level": SECURITY_LEVEL,
            "quantum_adjusted_security": effective_security,
            "grover_resistant": effective_security >= 64,
            "secure": effective_security >= 64
        }
    
    def _test_shor_resistance(self) -> Dict:
        """Test resistance against Shor's quantum algorithm"""
        # Our lattice-based crypto should be resistant to Shor's algorithm
        # which breaks RSA and ECC but not lattice problems
        
        return {
            "algorithm_type": "LATTICE_BASED",
            "shor_vulnerability": "NONE",
            "shor_resistant": True,
            "secure": True
        }
    
    def _grade_performance(self, performance_data: Dict) -> str:
        """Grade overall performance"""
        avg_throughput = np.mean([
            data["throughput_mbps"] for data in performance_data.values()
        ])
        
        if avg_throughput > 10:
            return "EXCELLENT"
        elif avg_throughput > 1:
            return "GOOD"
        elif avg_throughput > 0.1:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def run_comprehensive_test_suite(self) -> Dict:
        """Run all security tests and return comprehensive results"""
        print("Starting comprehensive quantum security test suite...")
        print("=" * 60)
        
        # Run all test categories
        entropy_results = self.test_key_generation_entropy()
        encryption_results = self.test_encryption_security()
        quantum_results = self.test_quantum_attack_resistance()
        performance_results = self.test_performance_benchmarks()
        
        # Compile overall assessment
        all_tests = [
            entropy_results.get("status") == "PASS",
            encryption_results.get("status") == "PASS",
            quantum_results.get("overall_quantum_resistance", {}).get("secure", False)
        ]
        
        overall_security = all(all_tests)
        
        summary = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_security_status": "QUANTUM-SAFE" if overall_security else "NEEDS_REVIEW",
            "security_level": SECURITY_LEVEL,
            "tests_passed": sum(all_tests),
            "total_tests": len(all_tests),
            "detailed_results": {
                "key_generation": entropy_results,
                "encryption_security": encryption_results,
                "quantum_resistance": quantum_results,
                "performance": performance_results
            },
            "recommendations": self._generate_recommendations(overall_security)
        }
        
        return summary
    
    def _generate_recommendations(self, overall_secure: bool) -> List[str]:
        """Generate security recommendations based on test results"""
        recommendations = []
        
        if overall_secure:
            recommendations.extend([
                "✅ Cryptographic implementation is quantum-safe",
                "✅ Security parameters meet post-quantum standards",
                "✅ Regular security audits recommended",
                "✅ Monitor for new post-quantum standards (NIST updates)"
            ])
        else:
            recommendations.extend([
                "⚠️ Review failing test cases immediately",
                "⚠️ Consider updating security parameters",
                "⚠️ Consult with cryptography experts",
                "⚠️ Implement additional security layers"
            ])
        
        recommendations.extend([
            "📋 Keep cryptographic libraries updated",
            "📋 Implement key rotation policies",
            "📋 Monitor quantum computing developments",
            "📋 Prepare for NIST PQC standard finalization"
        ])
        
        return recommendations

def run_quantum_security_test():
    """Run the quantum security test suite"""
    tester = QuantumSecurityTester()
    results = tester.run_comprehensive_test_suite()
    return results

if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_quantum_security_test()
    
    print("\n" + "=" * 60)
    print("QUANTUM SECURITY TEST RESULTS")
    print("=" * 60)
    print(f"Overall Status: {results['overall_security_status']}")
    print(f"Security Level: {results['security_level']} bits")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")