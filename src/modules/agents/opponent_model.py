from torch import nn
import torch
import numpy as np

class AgentModel(nn.Module):
    def __init__(self, input_shape, reconstruction_dims, latent_dims=32):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(np.prod(input_shape.shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.decode_head1 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[0])))
        self.decode_head2 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[1]), nn.Softmax()))
        self.decode_head3 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[2])))
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        head1 = self.decode_head1(decoded)
        head2 = self.decode_head2(decoded)
        head3 = self.decode_head3(decoded)
        return head1, head2, head3
    
    def encoder(self, x):
        return self.encode(x)