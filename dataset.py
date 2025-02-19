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
    def __init__(self):
        super().__init__()
        self.data = load_dataset("nlphuji/flickr30k", cache_dir="./data", split="test")
        self.processor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = self.processor.tokenizer
        self.vocab_size = self.tokenizer.vocab_size



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
