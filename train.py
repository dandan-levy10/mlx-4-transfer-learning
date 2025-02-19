import torch
from tqdm import tqdm
from decoder import Decoder
from vocab_projection import VocabProjection
from clip_model import CLIPModel
from dataset import ImageCaptionDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from utils import DEVICE

def train_loop(dataloader, clip_model, decoder, vocab_projection, optimizer, scheduler, num_epochs: int= 1):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for batch_idx, batch in enumerate(progress_bar):
            # batch = next(iter(dataloader))
            images, captions = batch
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        clip_model.eval()
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values=images)
            text_features = clip_model.text_model(
                input_ids=captions,
                output_hidden_states=True
            )
            last_hidden_state = text_features.last_hidden_state  # shape: (batch size, sequence length, hidden size)
            # print(last_hidden_state.shape, image_features.shape)

            # Zero the gradients
            optimizer.zero_grad()

            # Concatenate the image features and the last hidden state
            decoder_input = torch.cat((image_features.unsqueeze(1), last_hidden_state), dim=1)

            # Pass the concatenated features to the decoder
            logits = decoder(decoder_input)

            # Project the logits to the vocabulary size
            logits = vocab_projection(logits)

            # Remove the CLS token
            logits = logits[:, 1:, :].contiguous()

            # Flatten the logits and targets
            logits = logits.view(-1, logits.size(-1))
            targets = captions.view(-1)

            # Compute the loss
            loss = nn.CrossEntropyLoss()(logits, targets)
            epoch_loss += loss.item()

            # Backpropagate the loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 25 == 0:
                tqdm.write(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss /= len(dataloader)
    tqdm.write(f"Epoch {epoch + 1} completed, Epoch loss: {epoch_loss:.4f}")