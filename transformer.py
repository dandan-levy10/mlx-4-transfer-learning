class Transformer(nn.Module):
    def forward(self, image, caption_ids):
        input_sequence = torch.cat([image_features, caption_features], dim=1)
        logits, _ = self.decoder(input_sequence)  # Ignore attention weights if not needed
        logits = self.vocab_projection(logits)
        return logits 