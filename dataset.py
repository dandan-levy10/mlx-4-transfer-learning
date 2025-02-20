import torch
import transformers
from datasets import load_dataset


# class FlickrDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         super().__init__()
#         self.data = load_dataset("nlphuji/flickr30k", cache_dir="./data", split="test")
#         self.processor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image = item["image"]
#         caption = item["caption"][0]  # select first caption
#         encoding = self.processor(
#             text=[caption],
#             images=image,
#             return_tensors="pt",
#             truncation=True,
#             max_length=77,
#             padding="max_length",
#             add_special_tokens=True,
#         )

#         # Remove the extra batch dimension so that we always return a fixed shape of [77]
#         image = encoding["pixel_values"].squeeze(0)
#         caption_ids = encoding["input_ids"].squeeze(0)
#         return image, caption_ids       
    

class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, max_length=77):
        super().__init__()
        self.data = load_dataset("nlphuji/flickr30k", cache_dir="./data", split="test")
        self.processor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_length

        # Ensure that the EOS token is distinct from the PAD token.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            special_tokens_dict = {"eos_token": "<EOS>"}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print("Added new EOS token. Now pad and eos are different.",
                  "pad:", self.tokenizer.pad_token_id,
                  "eos:", self.tokenizer.eos_token_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        caption = item["caption"][0]  # select first caption

        encoding = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=77,
            padding="max_length",
        )

        # Remove the extra batch dimension so that we always return a fixed shape.
        image = encoding["pixel_values"].squeeze(0)
        caption_ids = encoding["input_ids"].squeeze(0)

        return image, caption_ids

    def process_caption(self, caption: str):
        """
        Tokenize the caption (with padding) and then replace the first PAD token
        with EOS (or force EOS at the last position if no PAD is found).
        """
        encoding = self.tokenizer(caption, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encoding['input_ids']
        if self.tokenizer.pad_token_id in input_ids:
            first_pad = input_ids.index(self.tokenizer.pad_token_id)
            input_ids[first_pad] = self.tokenizer.eos_token_id
        else:
            input_ids[-1] = self.tokenizer.eos_token_id
        return input_ids
