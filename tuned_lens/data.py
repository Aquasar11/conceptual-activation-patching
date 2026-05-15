import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple

from config import TunedLensConfig


class ChunkedTextDataset(Dataset):
    """Tokenizes a list of texts and packs them into fixed-length chunks."""

    def __init__(self, texts, tokenizer, seq_len: int):
        # Tokenize all texts and concatenate into one long token sequence
        all_ids = []
        for text in texts:
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)

        # Split into fixed-length chunks; drop the remainder
        self.seq_len = seq_len
        num_chunks = len(all_ids) // seq_len
        all_ids = all_ids[: num_chunks * seq_len]
        self.chunks = torch.tensor(all_ids, dtype=torch.long).view(num_chunks, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def build_dataloaders(config: TunedLensConfig) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader)."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    raw = load_dataset(config.dataset_name, config.dataset_config)

    train_texts = raw["train"]["text"]
    val_texts = raw["validation"]["text"]

    train_dataset = ChunkedTextDataset(train_texts, tokenizer, config.seq_len)
    val_dataset = ChunkedTextDataset(val_texts, tokenizer, config.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    return train_loader, val_loader
