import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, apply_mask: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.apply_mask = apply_mask
        
        self.W_q = nn.Linear(in_features = embedding_dim, out_features = embedding_dim)
        nn.init.xavier_uniform_(self.W_q.weight) # Initialize the weights using Xavier uniform initialization
        self.W_k = nn.Linear(in_features = embedding_dim, out_features = embedding_dim)
        nn.init.xavier_uniform_(self.W_k.weight)
        self.W_v = nn.Linear(in_features = embedding_dim, out_features = embedding_dim)
        nn.init.xavier_uniform_(self.W_v.weight)
        self.linear = nn.Linear(in_features = embedding_dim, out_features= embedding_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.image_bias = nn.Parameter(torch.tensor(1.0)) # Adding a bias for the image token
    
    def forward(self, x: torch.Tensor) -> tuple:
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim) # reshape the tensor
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim) # reshape the tensor
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim) # reshape the tensor

        # Permute tensors to shape (batch_size, num_heads, sequence_length, head_dim)- swapping sequence_length and num_heads
        Q = torch.permute(Q, (0, 2, 1, 3))
        K = torch.permute(K, (0, 2, 1, 3))
        V = torch.permute(V, (0, 2, 1, 3))

        A = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        A[:, :, :, 0] += self.image_bias  # Boost attention to image token

        # Apply mask if apply_mask is True
        if self.apply_mask:
            # Create the lower-triangular mask directly on the same device as the input tensor
            mask = torch.tril(torch.ones(sequence_length, sequence_length, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            A = A.masked_fill(mask == 0, float('-inf'))

        # Apply softmax along rows, the keys dim
        A = torch.softmax(A, dim= -1) # (batch_size, num_heads, sequence_length, sequence_length)

        # Store raw attention weights
        self.attention_weights = A.detach().cpu()

        # Multiply attention weights @ V, concatenate along the 
        AV = (A @ V) # (batch_size, num_heads, sequence_length, head_dim)

        # Swap num_heads and sequence length
        AV = torch.permute(AV, (0,2,1,3)) # (batch_size, sequence_length, num_heads, head_dim)
        
        # Concatenate head_dim, to return to input shape
        AV = AV.contiguous().view(batch_size, sequence_length, -1) # (batch_size, sequence_length, embedding_dim)
        
        # Linear projection
        output = self.linear(AV) # (batch_size, sequence_length, embedding_dim)

        # Dropout
        output = self.dropout(output)

        return output, A.detach()  # Return both output and attention weights
    
