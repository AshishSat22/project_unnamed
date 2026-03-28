import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SecureCNN

def train_model():
    # Fix seed for reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Simple transform: ToTensor + standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST Dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = SecureCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 4 # Train for 4 epochs to get reasonable accuracy
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping is important for polynomial activations to prevent overflow
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data):05d}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
                
        # Testing phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"=== Epoch {epoch} Test Results ===")
        print(f"Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
        
    # Save the trained model weights
    torch.save(model.state_dict(), "secure_cnn.pth")
    print("Model saved to secure_cnn.pth")

if __name__ == "__main__":
    train_model()
