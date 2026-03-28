import torch
from model import SecureCNN
import tenseal as ts

class Server:
    def __init__(self, model_path):
        self.model = SecureCNN()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Precompute the linear matrix equivalents for the layers to perform 
        # matrix-vector multiplication in TenSEAL.
        with torch.no_grad():
            # 1. Conv layer (1x28x28 -> 4x8x8) mapped to (784 -> 256)
            # Extract bias first
            zero_img = torch.zeros(1, 1, 28, 28)
            conv_b = self.model.conv1(zero_img).view(256)
            
            # Pass identity matrix through conv to extract weights without bias
            I = torch.eye(784).view(784, 1, 28, 28)
            conv_w = self.model.conv1(I).view(784, 256) - conv_b.view(1, 256)
            
            # Store as transposed python lists because tenseal matmul expects matrix layout matching
            # vector.matmul(matrix) where vector is 1xN and matrix is NxM -> 1xM
            self.W_conv = conv_w.tolist() # shape (784, 256)
            self.b_conv = conv_b.tolist()
            
            # 2. FC layer
            fc_w = self.model.fc1.weight.t() # shape (256, 10)
            fc_b = self.model.fc1.bias
            
            self.W_fc = fc_w.tolist()
            self.b_fc = fc_b.tolist()
            
    def process(self, enc_image):
        """
        enc_image: CKKSVector of size 784
        Returns: CKKSVector of size 10 containing the prediction
        """
        # Linear transform for Conv1
        enc_out = enc_image.matmul(self.W_conv)
        enc_out = enc_out + self.b_conv
        
        # Polynomial activation (square)
        enc_out = enc_out.square()
        
        # Fully connected layer
        enc_out = enc_out.matmul(self.W_fc)
        enc_out = enc_out + self.b_fc
        
        return enc_out
