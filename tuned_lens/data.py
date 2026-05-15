import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Tuple

from config import TunedLensConfig


class ChunkedTextDataset(Dataset):
    """Tokenizes a list of texts and packs them into fixed-length chunks."""

    def __init__(self, texts, tokenizer, seq_len: int, desc: str = "Tokenizing"):
        texts = [t for t in texts if t.strip()]

        # Batch tokenization: the fast (Rust) tokenizer parallelizes across threads per batch
        all_ids = []
        batch_size = 2000
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, dynamic_ncols=True):
            encoded = tokenizer(texts[i : i + batch_size], add_special_tokens=False)["input_ids"]
            for ids in encoded:
                all_ids.extend(ids)

        # Split into fixed-length chunks; drop the remainder
        self.seq_len = seq_len
        num_chunks = len(all_ids) // seq_len
        self.chunks = torch.tensor(all_ids[: num_chunks * seq_len], dtype=torch.long).view(num_chunks, seq_len)

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

    train_dataset = ChunkedTextDataset(train_texts, tokenizer, config.seq_len, desc="Tokenizing train")
    val_dataset = ChunkedTextDataset(val_texts, tokenizer, config.seq_len, desc="Tokenizing val")

    print(f"Dataset ready: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks "
          f"(seq_len={config.seq_len})")

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
