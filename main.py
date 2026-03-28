import os
import sys

def run():
    print("==================================================")
    print(" Privacy-Preserving Image Classification Launcher ")
    print("==================================================\n")
    
    # 1. Ensure Model Exists (Auto-Load and Auto-Train)
    if not os.path.exists("secure_cnn.pth"):
        print("[!] Model weights 'secure_cnn.pth' not found. Heating up training pipeline...\n")
        import train
        train.train_model()
    else:
        print("[*] Model weights 'secure_cnn.pth' successfully located and loaded.")
        
    print("\n[*] Validating mathematically secure TenSEAL inference...")
    
    # 2. Run Pipeline Test (Silencing pytest requirement)
    import test_pipeline
    try:
        # Manually run the unit test seamlessly without external CLI
        tester = test_pipeline.TestEncryptedPipeline("test_forward_pass_encryption")
        tester.setUp()
        tester.test_forward_pass_encryption()
        print("[+] Cryptographic Pipeline Verification: PASSED (Precision safely bounds within noise limits)")
    except Exception as e:
        print(f"[-] Cryptographic Pipeline Verification: FAILED - {str(e)}")
        sys.exit(1)
        
    print("\n[*] Sprinting rigorous execution benchmarks...")
    
    # 3. Fire Benchmarks
    import benchmark
    benchmark.run_benchmark()

if __name__ == "__main__":
    run()
