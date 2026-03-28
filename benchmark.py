import torch
import time
import tenseal as ts
from model import SecureCNN
from he_context import create_context
from client import Client
from server import Server
from torchvision import datasets, transforms
import os

def run_benchmark():
    device = torch.device('cpu')
    model_path = "secure_cnn.pth"
    if not os.path.exists(model_path):
        print("Model file 'secure_cnn.pth' not found. Please run train.py first to generate the weights.")
        return
        
    # Setup context and instances
    context = create_context()
    client = Client(context)
    server = Server(model_path)
    
    # Load model for plaintext inference
    model = SecureCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load small subset of MNIST test data to measure accuracy
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Subset of 50 images to keep HE evaluation time reasonable yet robust
    subset_indices = list(range(50))
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, subset_indices), 
        batch_size=1, shuffle=False
    )
    
    print("--- Running Benchmarks ---")
    correct_plain = 0
    correct_enc = 0
    latencies = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            # Plaintext inference
            plain_pred = model(data)
            plain_label = plain_pred.argmax(dim=1, keepdim=True).item()
            if plain_label == target.item():
                correct_plain += 1
                
            # Encrypted inference
            t0 = time.time()
            enc_image = client.encrypt_image(data)
            t1 = time.time()
            
            # This is the measured inference latency
            enc_pred = server.process(enc_image)
            t2 = time.time()
            
            dec_pred = client.decrypt_prediction(enc_pred)
            t3 = time.time()
            
            enc_label = dec_pred.argmax(dim=1, keepdim=True).item()
            if enc_label == target.item():
                correct_enc += 1
                
            latency_ms = (t2 - t1) * 1000
            latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies)
    plain_acc = (correct_plain / 50) * 100
    enc_acc = (correct_enc / 50) * 100
    
    print("\n================ Benchmark Report ================")
    print(f"Number of test images     : 50")
    print(f"Plaintext Accuracy        : {plain_acc:.2f}%")
    print(f"Encrypted Accuracy        : {enc_acc:.2f}%")
    print(f"Average Server Latency    : {avg_latency:.2f} ms")
    print("==================================================")
    
    if avg_latency < 200:
        print("[PASS] Target inference latency < 200ms achieved!")
    else:
        print("[FAIL] Inference latency exceeded 200ms target.")
        
if __name__ == "__main__":
    run_benchmark()
