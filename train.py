import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from transformers import CLIPModel
from dataset import FlickrDataset
from decoder import Decoder, Transformer
from utils import DEVICE
from helpers import insert_eos_token, print_sample_prediction

def get_embeddings(clip_model, images, captions, tokenizer):
    clip_model.eval()
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=images)
        text_features = clip_model.text_model(
            input_ids=captions,
            output_hidden_states=True
        )
        text_features = text_features.last_hidden_state  # (batch, seq_len, hidden_dim)
        combined_features = torch.cat((image_features.unsqueeze(1), text_features), dim=1)
        return combined_features

def calculate_loss(logits, targets, tokenizer, loss_fn):
    # # Temporary debug
    # print("Input captions sample:", tokenizer.decode(targets[0].tolist()))
    # print("Trimmed logits shape:", logits[:, :-1, :].shape)
    # print("Shifted targets shape:", targets[:, 1:].shape)
    
    # # Visualize token positions
    # plt.matshow(targets[0].cpu().numpy()[:, None] == targets[0].cpu().numpy())
    # plt.title("Token Position Alignment")
    # plt.show()
    
    # Remove the final token from logits (based on <EOS> token)
    logits = logits[:, :-1, :].contiguous()  # shape: (batch, seq_len, vocab_size)
    # Flatten the logits and targets for loss computation.
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    # Use the tokenizer's pad token as ignore_index (if provided)
    ignore_index = tokenizer.pad_token_id if tokenizer is not None else -100
    loss = loss_fn(logits, targets)
    # print("Logits range:", logits.min().item(), logits.max().item())
    # print("Logits mean:", logits.mean().item())
    return loss
    
def calculate_accuracy(logits, targets, tokenizer):
    """
    Computes token-level accuracy based on raw logits.
    Expects logits of shape (B, seq_len, vocab_size); it discards the prediction for the final token,
    computes predicted token IDs via argmax, and compares them against the target tokens while ignoring padding.
    
    Args:
        logits (torch.Tensor): Raw logits with shape (batch, sequence_length, vocab_size).
        targets (torch.Tensor): Target token IDs with shape (batch, sequence_length).
        tokenizer: The tokenizer, used to obtain the pad_token_id.
    
    Returns:
        float: Accuracy as a fraction between 0 and 1.
    """
    # Remove the final token from logits to match the target sequence length.
    trimmed_logits = logits[:, :-1, :].contiguous()  # shape: (B, seq_len-1, vocab_size)
    # Compute predictions by taking the argmax over the last dimension.
    pred_ids = trimmed_logits.argmax(dim=-1)  # shape: (B, seq_len-1)
    # Flatten predictions and targets for a fair, elementwise comparison.
    pred_ids = pred_ids.view(-1)
    targets_flat = targets.view(-1)
    # Create a mask to ignore positions with the pad token.
    mask = targets_flat != tokenizer.pad_token_id
    # Calculate the number of correct predictions.
    correct = (pred_ids[mask] == targets_flat[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def calculate_accuracy_counts(logits, targets, tokenizer):
    # Trim logits so they align with the targets.
    trimmed_logits = logits[:, :-1, :].contiguous() 
    pred_ids = trimmed_logits.argmax(dim=-1)           
    pred_ids = pred_ids.view(-1)
    targets_flat = targets.view(-1)
    mask = targets_flat != tokenizer.pad_token_id
    correct = (pred_ids[mask] == targets_flat[mask]).sum().item()
    total = mask.sum().item()
    return correct, total

def train_loop(dataloader, transformer, optimizer, scheduler, num_epochs, loss_fn, tokenizer=None):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch_idx, (images, captions) in enumerate(progress_bar):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = transformer(images, captions)
            loss = calculate_loss(logits, captions, tokenizer, loss_fn)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Batch accuracy calculation (using the new function).
            correct, total = calculate_accuracy_counts(logits, captions, tokenizer)
            epoch_correct += correct
            epoch_total += total

            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                batch_acc = correct / total if total > 0 else 0.0
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_acc*100:.2f}%\n")
                if tokenizer is not None:
                    print_sample_prediction(logits, captions, tokenizer)

            # Temporary addition
            # print("Image projection grad norm:", 
            #       transformer.projection[0].weight.grad.norm().item())
            # print("Decoder first layer grad norm:",
            #       transformer.decoder.layers[0].attention.q_linear.weight.grad.norm().item())

            # After backward()
            # print("Grad norms:")
            # for name, param in transformer.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item():.4f}")

        # End of epoch: compute epoch-level accuracy.
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        tqdm.write(f"Epoch {epoch+1} completed, Epoch Loss: {epoch_loss/len(dataloader):.4f}, Epoch Accuracy: {epoch_accuracy*100:.2f}%")

def run_training(num_epochs=1, batch_size=64, lr=0.001,
                 num_layers=1, embedding_dim=512, num_heads=2, ff_dim=1024,
                 step_size=5, gamma=0.5):
    """
    This function initializes all components (dataset, model, optimizer, etc.)
    and then runs the training loop.
    
    Arguments:
        num_epochs: number of epochs to train.
        batch_size: batch size for the data loader.
        lr: learning rate for the optimizer.
        num_layers: number of decoder layers.
        embedding_dim: embedding dimension of the decoder.
        num_heads: number of heads in multi-head attention.
        ff_dim: feed-forward network dimension.
        step_size: stepsize for the learning rate scheduler.
        gamma: LR decay rate.
    """
    # 1. Initialize dataset and model
    dataset = FlickrDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    transformer = Transformer(embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, dropout=0.1, max_seq_len=78, apply_mask=True, input_dim=512, output_dim=512, device=DEVICE).to(DEVICE)
    
    # 2. Simple loss with label smoothing
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=dataset.tokenizer.pad_token_id,
        label_smoothing=0.1
    )
    
    # 3. Optimizer with gradient clipping
    optimizer = optim.AdamW(transformer.parameters(), lr=lr)

    # 4. Set up the OneCycleLR scheduler.
    total_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Save the state dictionaries after training.
    checkpoint = {
        'transformer_state': transformer.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hyperparameters': {
            'num_layers': num_layers,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'step_size': step_size,
            'gamma': gamma
        }
    }

    # 5. Training loop
    train_loop(dataloader, transformer, optimizer, scheduler, num_epochs, loss_fn, tokenizer=dataset.tokenizer)
    torch.save(checkpoint, "model_checkpoint.pth")
    print("Saved checkpoint to model_checkpoint.pth")

if __name__ == "__main__":
    run_training(num_epochs=3)  # Modify hyperparameters as needed.