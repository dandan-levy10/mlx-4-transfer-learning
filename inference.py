import torch
from utils import DEVICE

from dataset import FlickrDataset
from decoder import Decoder
from utils import DEVICE
from transformers import CLIPModel
from transformers import AutoTokenizer


def test_model(dataloader, clip_model, decoder, vocab_projection, tokenizer, num_examples=5):
    """
    Runs the model on some examples from the dataloader and returns a list 
    of tuples: (predicted caption, ground-truth caption).

    This uses the same teacher-forcing style as in training. For autoregressive
    generation you would need to loop and feed the generated token back.
    """
    clip_model.eval()
    results = []
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            
            # Use the CLIP model to extract features (as in training)
            image_features = clip_model.get_image_features(pixel_values=images)
            text_features = clip_model.text_model(
                input_ids=captions,
                output_hidden_states=True
            )
            last_hidden_state = text_features.last_hidden_state  # (batch, seq_len, hidden_dim)
            
            # Prepare decoder input by concatenating image features as a CLS token with text features.
            decoder_input = torch.cat((image_features.unsqueeze(1), last_hidden_state), dim=1)
            logits = decoder(decoder_input)
            logits = vocab_projection(logits)
            
            # Remove the image/CLS token output at index 0 to match caption targets.
            logits = logits[:, 1:, :].contiguous()  # (batch, seq_len, vocab_size)
            predictions = logits.argmax(dim=-1)      # (batch, seq_len)
            
            # For each example in the batch decode predictions and targets.
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i].tolist()
                true_tokens = captions[i].tolist()
                # Use the tokenizer to decode. Skip special tokens if desired.
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                true_text = tokenizer.decode(true_tokens, skip_special_tokens=True)
                results.append((pred_text, true_text))
                if len(results) >= num_examples:
                    return results
    return results


def generate_caption(image, clip_model, decoder, vocab_projection, tokenizer, max_length=77):
    """
    Generates a caption for a single image via free-running autoregressive inference.
    
    Args:
        image (torch.Tensor): A single image tensor (already preprocessed) with shape (C, H, W).
        clip_model (CLIPModel): The frozen CLIP model used to extract image features.
        decoder (Decoder): The decoder model used to generate token embeddings.
        vocab_projection (nn.Module): Linear layer projecting to vocabulary size.
        tokenizer: A tokenizer (or processor.tokenizer) that maps tokens to IDs and can decode them.
        max_length (int): Maximum number of tokens to generate.
        
    Returns:
        caption (str): The generated caption string.
    """
    # Put the models in evaluation mode.
    clip_model.eval()
    decoder.eval()

    with torch.no_grad():
        # 1. Extract image features.
        # Note: image should be shaped appropriately (e.g. [C, H, W]); we add a batch dimension.
        image_feature = clip_model.get_image_features(pixel_values=image.unsqueeze(0).to(DEVICE))
        # 2. Get the <start> token ID.
        start_token_id = tokenizer.convert_tokens_to_ids("<start>")
        generated = [start_token_id]
        
        # 3. Loop to generate tokens autoregressively.
        for _ in range(max_length):
            # Convert the current sequence to a tensor of shape (1, seq_len).
            current_seq = torch.tensor(generated, device=DEVICE).unsqueeze(0)  # (1, seq_len)
            # Get token embeddings using the CLIP text model's embedding layer.
            token_embeddings = clip_model.text_model.embeddings(current_seq)  # (1, seq_len, embedding_dim)
            # Concatenate image features as the first "token" (i.e. CLS token) with text embeddings.
            # The decoder is expecting an input sequence with image feature followed by text tokens.
            decoder_input = torch.cat([image_feature.unsqueeze(1), token_embeddings], dim=1)  # (1, 1+seq_len, embedding_dim)
            # Forward pass through the decoder.
            logits = decoder(decoder_input)
            # Project the decoder's output to vocabulary logits.
            logits = vocab_projection(logits)
            # Remove the decoder's output for the image token (first position).
            logits = logits[:, 1:, :]  # (1, seq_len, vocab_size)
            # Get the logits for the last time-step (the most recent generated token).
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            # Choose the token with highest probability (argmax). You could also sample here.
            temperature = 1.0  # you can adjust this value (lower for more deterministic, higher for more diverse)
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            # Append the predicted token ID.
            generated.append(next_token_id)
            # Stop if <end> token is generated.
            if next_token_id == tokenizer.convert_tokens_to_ids("<end>"):
                break

        # Decode the generated token IDs into a string.
        # Optionally, skip special tokens to get a cleaner output.
        caption = tokenizer.decode(generated, skip_special_tokens=True)
    return caption


# Example usage if running this file directly:
if __name__ == "__main__":
    # Create a dataloader for inference. Here we simply use the first image from your dataset.
    dataset = FlickrDataset()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    
    # Load your models and tokenizer. (Adjust these instantiations as in your training script.)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    # Freeze CLIP parameters.
    for param in clip_model.vision_model.parameters():
        param.requires_grad = False
    for param in clip_model.text_model.parameters():
        param.requires_grad = False

    # Inference requires the decoder and vocab projection.
    import torch.nn as nn
    try:
        checkpoint = torch.load("model_checkpoint.pth", map_location=DEVICE)
        # Extract hyperparameters from the checkpoint
        hp = checkpoint.get('hyperparameters', {})
        num_layers = hp.get('num_layers', 2)
        embedding_dim = hp.get('embedding_dim', 128)
        num_heads = hp.get('num_heads', 4)
        ff_dim = hp.get('ff_dim', 256)
    
        # Reinitialize the decoder and vocab projection using checkpoint hyperparameters.
        decoder = Decoder(num_layers=num_layers, embedding_dim=embedding_dim, 
                          num_heads=num_heads, ff_dim=ff_dim).to(DEVICE)
        vocab_projection = nn.Linear(embedding_dim, dataset.vocab_size).to(DEVICE)
    
        # Load the saved model states.
        decoder.load_state_dict(checkpoint['decoder_state'])
        vocab_projection.load_state_dict(checkpoint['vocab_projection_state'])
        print("Model checkpoint loaded successfully.")
    except FileNotFoundError:
        print("No checkpoint found. Using randomly initialized weights.")
    
    # Get the tokenizer. For example, from your dataset's processor:
    tokenizer = dataset.tokenizer
    
    import matplotlib.pyplot as plt
    # Get one image from the dataloader.
    for images, gt_captions in dataloader:
        # Take the first image and its ground-truth caption from the batch.
        image = images[0]
        gt_caption = tokenizer.decode(gt_captions[0].tolist(), skip_special_tokens=True)
        gen_caption = generate_caption(image, clip_model, decoder, vocab_projection, tokenizer)

        # Unnormalize the image for display purposes.
        # CLIP normalization values; adjust if you've used different ones.
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(DEVICE)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(DEVICE)
        # Move the image to DEVICE so that all tensors are on the same device.
        img_unnorm = image.to(DEVICE) * std[:, None, None] + mean[:, None, None]
        img_unnorm = img_unnorm.clamp(0, 1).cpu().permute(1, 2, 0).numpy()

        # Display the image with the ground-truth and generated captions.
        plt.imshow(img_unnorm)
        plt.title(f"Ground Truth: {gt_caption}\nGenerated: {gen_caption}")
        plt.axis("off")
        plt.show()
