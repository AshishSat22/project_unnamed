import torch
import unittest
import tenseal as ts
import os
from model import SecureCNN
from he_context import create_context
from client import Client
from server import Server

class TestEncryptedPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary model file and dummy weights for testing
        self.model_path = "temp_test_secure_cnn.pth"
        model = SecureCNN()
        torch.save(model.state_dict(), self.model_path)
        
        self.context = create_context()
        self.client = Client(self.context)
        self.server = Server(self.model_path)
        
        self.model = model
        self.model.eval()

    def test_forward_pass_encryption(self):
        # Generate a random dummy image (1x28x28) scaled properly
        torch.manual_seed(42)
        dummy_image = torch.rand(1, 1, 28, 28)
        
        # Plaintext prediction
        with torch.no_grad():
            plain_pred = self.model(dummy_image)
            
        # Encrypted prediction
        enc_image = self.client.encrypt_image(dummy_image)
        enc_pred = self.server.process(enc_image)
        decrypted_pred = self.client.decrypt_prediction(enc_pred)
        
        # Compare
        difference = (plain_pred - decrypted_pred).abs().max().item()
        print(f"\nMaximum absolute difference between Plain and Encrypted: {difference}")
        
        # Test tolerance 0.01
        self.assertTrue(torch.allclose(plain_pred, decrypted_pred, atol=0.01))

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
