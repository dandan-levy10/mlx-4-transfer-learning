import torch
from transformers import AutoTokenizer

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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print(insert_eos_token(torch.tensor([[1, 2, 3, 4, 5]]), tokenizer))
    print(insert_eos_token(torch.tensor([[1, 2, 3, 4, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id]]), tokenizer))
    print(tokenizer.pad_token_id, tokenizer.eos_token_id)