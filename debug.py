import torch
import json
from test_pipeline import TestEncryptedPipeline

def debug():
    tester = TestEncryptedPipeline("test_forward_pass_encryption")
    tester.setUp()
    
    torch.manual_seed(42)  # matching the test seed
    dummy_image = torch.rand(1, 1, 28, 28)
    
    with torch.no_grad():
        plain_pred = tester.model(dummy_image)
        
    enc_image = tester.client.encrypt_image(dummy_image)
    enc_pred = tester.server.process(enc_image)
    dec_pred = tester.client.decrypt_prediction(enc_pred)
    
    diff = (plain_pred - dec_pred).abs().max().item()
    
    result = {
        "plain": plain_pred.tolist(),
        "decrypted": dec_pred.tolist(),
        "diff": diff
    }
    
    with open("debug_results.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Debug finished with max difference {diff}")

if __name__ == "__main__":
    debug()
