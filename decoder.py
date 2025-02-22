import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import Attention
from mlp import MultiLayerPerceptron
import transformers

from utils import DEVICE
from helpers import insert_eos_token, print_sample_prediction

class Decoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, ff_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, max_seq_len: int = 100, apply_mask: bool = True, input_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.apply_mask = apply_mask
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.projection = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, ff_dim, dropout, apply_mask) for _ in range(num_layers)])
        self.output_projection = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.projection(x)
        all_attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attention_weights.append(attn_weights)
            
        x = self.output_projection(x)
        return x, all_attention_weights
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, apply_mask: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = Attention(embedding_dim, num_heads, dropout, apply_mask=apply_mask)
        self.mlp = MultiLayerPerceptron(embedding_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        # Attention block
        residual = x
        x = self.norm1(x)
        x_attn, attn_weights = self.attention(x)
        x = residual + x_attn
        
        # MLP block
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x, attn_weights

# class Attention(nn.Module):
#     def __init__(self, embedding_dim, num_heads, dropout=0.1, apply_mask=True):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.apply_mask = apply_mask
#         self.return_attention = False
        
#         assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
#         self.head_dim = embedding_dim // num_heads
        
#         self.q_linear = nn.Linear(embedding_dim, embedding_dim)
#         nn.init.xavier_uniform_(self.q_linear.weight)
#         self.k_linear = nn.Linear(embedding_dim, embedding_dim)
#         nn.init.xavier_uniform_(self.k_linear.weight)
#         self.v_linear = nn.Linear(embedding_dim, embedding_dim)
#         nn.init.xavier_uniform_(self.v_linear.weight)
#         self.out_linear = nn.Linear(embedding_dim, embedding_dim)
#         nn.init.xavier_uniform_(self.out_linear.weight)
        
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
        
#         # Linear projections and reshape for multi-head
#         q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Scaled dot-product attention
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
#         # Apply causal mask if specified
#         if self.apply_mask:
#             mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
#             scores.masked_fill_(mask, float('-inf'))
        
#         attention_weights = F.softmax(scores, dim=-1)
#         attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
#         # Apply attention weights to values
#         out = torch.matmul(attention_weights, v)
        
#         # Reshape and apply output projection
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
#         out = self.out_linear(out)
        
#         return out if not self.return_attention else (out, attention_weights)

class Transformer(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_heads: int = 8, ff_dim: int = 1024, num_layers: int = 12, dropout: float = 0.1, max_seq_len: int = 77, apply_mask: bool = True, input_dim: int = 512, output_dim: int = 512, device: str = DEVICE):
        super().__init__()
        self.clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.embeddings = self.clip_model.text_model.embeddings
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, dropout=dropout, max_seq_len=max_seq_len, apply_mask=apply_mask, input_dim=input_dim, output_dim=output_dim)
        self.vocab_projection = nn.Linear(embedding_dim, self.embeddings.token_embedding.weight.shape[0], bias=False)
        with torch.no_grad():
            self.vocab_projection.weight.data.copy_(self.embeddings.token_embedding.weight.data)
        self.clip_model.requires_grad_(False)  # freeze CLIP model
        self.embeddings.requires_grad_(False)  # freeze text embeddings

    def forward(self, image, caption_ids):
        image_features = self.clip_model.get_image_features(pixel_values=image)
        image_features = F.normalize(image_features, dim=-1)        
        image_features = self.projection(image_features)
        # Normalize the image features after projection to ensure stable training.
        image_features = F.normalize(image_features, p=2, dim=-1)
        image_features = image_features.unsqueeze(1)
        caption_features = self.embeddings(caption_ids)
        input_sequence = torch.cat([image_features, caption_features], dim=1)
        logits, attention_weights = self.decoder(input_sequence)
        logits = self.vocab_projection(logits)
        return logits, attention_weights
    
    # def predict(self, image, caption_ids):
    #     logits = self.forward(image, caption_ids)
    #     prd = torch.argmax(logits, dim=-1)
    #     prd = torch.multinomial(torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1)
    #     return torch.cat([caption_ids, prd], dim=1)


def test_masking():
    """
    Simple test to verify that the attention masking is working correctly.
    """
    # Create a single Attention layer for testing
    attention = Attention(
        embedding_dim=64,
        num_heads=4,
        dropout=0.1,
        apply_mask=True
    ).to(DEVICE)
    
    # Create a sample input sequence
    batch_size = 1
    seq_len = 5
    x = torch.randn(batch_size, seq_len, 64).to(DEVICE)
    
    # Set return_attention to True to get the weights
    attention.return_attention = True
    
    # Forward pass
    with torch.no_grad():
        _, attention_weights = attention(x)
    
    # Get the attention weights from the output
    weights = attention_weights.squeeze(0)  # Remove batch dimension
    print("Attention weight shape:", weights.shape)
    
    # The upper triangle should be close to zero if masking is working
    upper_triangle = torch.triu(weights[0], diagonal=1)  # Take first head
    print("\nUpper triangle (should be close to 0):")
    print(upper_triangle)
    
    # Check if upper triangle is close to zero
    is_masked = torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-8)
    print("\nMasking working correctly:", is_masked)
    
    # Visualize the attention pattern
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(weights[0].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights\n(Upper triangle should be zero)')
    plt.show()

if __name__ == "__main__":
    test_masking()
    test_masking()