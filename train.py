import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import CLIPModel
from dataset import FlickrDataset
from decoder import Decoder
from utils import DEVICE

def print_sample_prediction(logits_3d, targets_2d, tokenizer):
    """
    Given unflattened logits (shape: [batch, seq_len, vocab_size]) and targets (shape: [batch, seq_len]),
    decode the first sample in the batch and print the predicted vs. target text.
    """
    # Get sample predictions for the first sample in the batch.
    sample_logits = logits_3d[0]  # shape: (seq_len, vocab_size)
    sample_targets = targets_2d[0]  # shape: (seq_len)

    # Get predicted token ids via argmax
    pred_ids = sample_logits.argmax(dim=-1).tolist()
    true_ids = sample_targets.tolist()

    # Decode using the tokenizer (skip special tokens for readability)
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    true_text = tokenizer.decode(true_ids, skip_special_tokens=True)

    # Remove pad tokens from the target (and corresponding predictions)
    pad_id = tokenizer.pad_token_id
    filtered_pred_ids = [x for x, y in zip(pred_ids, true_ids) if y != pad_id]
    filtered_true_ids = [y for y in true_ids if y != pad_id]
    if len(filtered_true_ids) > 0:
        correct_predictions = sum(x == y for x, y in zip(filtered_pred_ids, filtered_true_ids))
        sample_accuracy = correct_predictions / len(filtered_true_ids) * 100
    else:
        sample_accuracy = 0.0
    print(f"Sample Accuracy  : {sample_accuracy:.2f}%")
    print()

    print("Sample Prediction:", pred_text)
    print("Sample Target    :", true_text)

def train_loop(dataloader, clip_model, decoder, vocab_projection, optimizer, scheduler, num_epochs, tokenizer=None):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_correct = 0
        total_count = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch_idx, (images, captions) in enumerate(progress_bar):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # Use the CLIP model to extract image and text features.
            clip_model.eval()
            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=images)
                text_features = clip_model.text_model(
                    input_ids=captions,
                    output_hidden_states=True
                )
                last_hidden_state = text_features.last_hidden_state  # (batch, seq_len, hidden_dim)

            optimizer.zero_grad()

            # Prepare decoder input by concatenating image features (as a CLS token)
            # with text features.
            decoder_input = torch.cat((image_features.unsqueeze(1), last_hidden_state), dim=1)
            logits = decoder(decoder_input)

            # Project the decoder's output to the vocabulary size.
            logits = vocab_projection(logits)

            # Remove the image/CLS token output to match caption targets.
            # Here we assume that the decoder inputs are [CLS, token1, token2, ...] so that we predict the caption tokens.
            # If captions is of shape (batch, seq_len), we extract logits corresponding to those positions.
            logits = logits[:, :-1, :].contiguous()  # shape: (batch, seq_len, vocab_size)
            # Save a copy of logits (unflattened) for sample printing.
            sample_logits_3d = logits.detach().cpu()
            # Targets remain as-is (assuming they are aligned with logits' time dimension)
            targets_2d = captions  # shape: (batch, seq_len)

            # Then flatten for loss computation.
            logits = logits.view(-1, logits.size(-1))
            targets = captions.view(-1)

            # Use the tokenizer's pad token as ignore_index (if provided)
            ignore_index = tokenizer.pad_token_id if tokenizer is not None else -100
            loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(logits, targets)
            
            # --- Accuracy computation added here ---
            predictions = logits.argmax(dim=-1)
            # Create a mask to ignore positions with the pad token.
            mask = targets != ignore_index
            batch_correct = (predictions == targets).masked_select(mask).sum().item()
            batch_total = mask.sum().item()
            total_correct += batch_correct
            total_count += batch_total
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0.0
            # ----------------------------------------
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy*100:.2f}%")
                # If a tokenizer is provided, print a sample prediction
                if tokenizer is not None:
                    print_sample_prediction(sample_logits_3d, targets_2d, tokenizer)

        # Move the scheduler step outside the batch loop:
        scheduler.step()
        epoch_accuracy = total_correct / total_count if total_count > 0 else 0.0
        tqdm.write(f"Epoch {epoch+1} completed, Epoch loss: {epoch_loss/len(dataloader):.4f}, Epoch Accuracy: {epoch_accuracy*100:.2f}%")
    

    
def run_training(num_epochs=1, batch_size=32, lr=0.001,
                 num_layers=2, embedding_dim=128, num_heads=4, ff_dim=256,
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
    # 1. Initialize the dataset and dataloader.
    dataset = FlickrDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Load the CLIP model.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    # Freeze CLIP model parameters.
    for param in clip_model.vision_model.parameters():
        param.requires_grad = False
    for param in clip_model.text_model.parameters():
        param.requires_grad = False

    # 3. Initialize the decoder.
    decoder = Decoder(num_layers=num_layers, embedding_dim=embedding_dim,
                      num_heads=num_heads, ff_dim=ff_dim).to(DEVICE)

    # 4. Initialize the vocabulary projection layer.
    vocab_projection = nn.Linear(embedding_dim, dataset.vocab_size).to(DEVICE)

    # 5. Create the optimizer (only updating decoder and vocab_projection parameters).
    optimizer = optim.Adam(list(decoder.parameters()) + list(vocab_projection.parameters()), lr=lr)

    # 6. Set up a learning rate scheduler.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 7. Run the training loop.
    train_loop(dataloader, clip_model, decoder, vocab_projection, optimizer, scheduler, num_epochs, tokenizer=dataset.tokenizer)

    # Save the state dictionaries after training.
    checkpoint = {
        'decoder_state': decoder.state_dict(),
        'vocab_projection_state': vocab_projection.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hyperparameters': {
            'num_layers': num_layers,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
        }
    }
    torch.save(checkpoint, "model_checkpoint.pth")
    print("Saved checkpoint to model_checkpoint.pth")

if __name__ == "__main__":
    run_training(num_epochs=1)  # Modify hyperparameters as needed.