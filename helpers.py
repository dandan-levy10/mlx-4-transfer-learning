import torch
from transformers import AutoTokenizer
from torch.nn.functional import cosine_similarity
from dataset import FlickrDataset
import transformers

from utils import DEVICE

def insert_eos_token(captions: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    For each caption in the batch (shape: [batch, seq_len]), replace the first PAD token
    with the EOS token. If no PAD token is found, replace the last token with EOS.

    Args:
        captions (torch.Tensor): A tensor of shape [batch, seq_len] containing token IDs.
        tokenizer: The tokenizer used to obtain eos_token_id and pad_token_id.

    Returns:
        torch.Tensor: The modified captions tensor with EOS inserted in the proper position.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    for i in range(captions.size(0)):
        # Find indices where the caption equals the PAD token.
        pad_positions = (captions[i] == pad_id).nonzero(as_tuple=True)[0]
        if pad_positions.numel() > 0:
            # Replace the first PAD token with EOS.
            captions[i, pad_positions[0]] = eos_id
        else:
            # If no padding is found, force EOS at the last index.
            captions[i, -1] = eos_id
    return captions

def print_sample_prediction(logits, captions, tokenizer):
    """
    Given unflattened logits (shape: [batch, seq_len, vocab_size]) and targets (shape: [batch, seq_len]),
    decode the first sample in the batch and print the predicted vs. target text.
    """

    logits_3d = logits[:, :-1, :].detach().cpu() # shape: (batch, seq_len, vocab_size)
    targets_2d = captions.detach().cpu() # shape: (batch, seq_len)

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

def validate_clip_embeddings(dataset, clip_model, num_samples=100):
    """Compute pairwise similarity of CLIP image embeddings."""
    clip_model = clip_model.eval().to(DEVICE)
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    embeddings = []
    for idx in indices:
        image, _ = dataset[idx]
        with torch.no_grad():
            # Use CLIP's image encoder (ensure you're extracting embeddings correctly)
            image_embed = clip_model.get_image_features(image.unsqueeze(0).to(DEVICE))
        embeddings.append(image_embed)
    embeddings = torch.cat(embeddings, dim=0)
    # Compute pairwise similarity
    sim_matrix = cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    print(f"Mean pairwise similarity: {sim_matrix.mean().item():.3f}")

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


if __name__ == "__main__":
    # Usage (assuming `clip_img_encoder` is your frozen CLIP image encoder):
    validate_clip_embeddings(dataset=FlickrDataset(), clip_model=transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE))
