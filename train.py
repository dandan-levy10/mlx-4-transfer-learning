import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import CLIPModel
from dataset import FlickrDataset
from decoder import Decoder
from utils import DEVICE

def train_loop(dataloader, clip_model, decoder, vocab_projection, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
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

            # Remove the image/CLS token output at index 0 to match caption targets.
            logits = logits[:, 1:, :].contiguous()
            logits = logits.view(-1, logits.size(-1))
            targets = captions.view(-1)

            loss = nn.CrossEntropyLoss()(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Remove scheduler.step() from here!
            epoch_loss += loss.item()

            if batch_idx % 25 == 0:
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # Move the scheduler step outside the batch loop:
        scheduler.step()
        tqdm.write(f"Epoch {epoch+1} completed, Epoch loss: {epoch_loss/len(dataloader):.4f}")
    

    
def run_training(num_epochs=1, batch_size=32, lr=0.001,
                 num_layers=1, embedding_dim=64, num_heads=4, ff_dim=128,
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
    train_loop(dataloader, clip_model, decoder, vocab_projection, optimizer, scheduler, num_epochs)

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