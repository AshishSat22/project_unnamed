import tenseal as ts
import torch

class Client:
    def __init__(self, context):
        self.context = context
        
    def encrypt_image(self, image_tensor):
        """
        Encrypts a 1x28x28 PyTorch tensor image into a TenSEAL CKKS vector.
        """
        # Flatten the image to perfectly match our linear matrix equivalent of Conv2d
        flat_image = image_tensor.view(-1).tolist()
        enc_image = ts.ckks_vector(self.context, flat_image)
        return enc_image
        
    def decrypt_prediction(self, enc_prediction):
        """
        Decrypts a TenSEAL CKKS vector prediction back into a PyTorch tensor.
        """
        decrypted_list = enc_prediction.decrypt()
        return torch.tensor(decrypted_list).view(1, -1)
