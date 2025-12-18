import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from typing import Any
from datasets import Dataset as HFDataset

class TranslationDataset(Dataset):
    def __init__(
        self, 
        dataset: HFDataset | dict[str, Any],
        src_sp: spm.SentencePieceProcessor, 
        tgt_sp: spm.SentencePieceProcessor,
        seq_len: int = 100
    ):
        self.dataset = dataset
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.seq_len = seq_len

        self.src_pad_id = src_sp.pad_id()
        self.tgt_pad_id = tgt_sp.pad_id()
        self.src_bos_id = src_sp.bos_id()
        self.tgt_bos_id = tgt_sp.bos_id()
        self.src_eos_id = src_sp.eos_id()
        self.tgt_eos_id = tgt_sp.eos_id()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoder_input_tokens = item["src_ids"]
        decoder_input_tokens = item["tgt_ids"]

        # Add BOS, EOS and padding tokens
        encoder_input = torch.cat(
            [
                torch.tensor(encoder_input_tokens, dtype=torch.long),
                torch.tensor([self.src_eos_id], dtype=torch.long),
                torch.tensor([self.src_pad_id] * (self.seq_len - len(encoder_input_tokens) - 1), dtype=torch.long),
            ],
            dim=0,
        )

        decoder_input = torch.cat([
            torch.tensor([self.tgt_bos_id], dtype=torch.long),
            torch.tensor(decoder_input_tokens, dtype=torch.long),
            torch.tensor([self.tgt_pad_id] * (self.seq_len - len(decoder_input_tokens) - 1), dtype=torch.long)
        ])

        label = torch.cat([
            torch.tensor(decoder_input_tokens, dtype=torch.long),
            torch.tensor([self.tgt_eos_id], dtype=torch.long),
            torch.tensor([self.tgt_pad_id] * (self.seq_len - len(decoder_input_tokens) - 1), dtype=torch.long)
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.src_pad_id).unsqueeze(0).unsqueeze(0), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.tgt_pad_id).unsqueeze(0).unsqueeze(1) & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
        }

def causal_mask(size):
    """
    Create a causal mask for decoder attention.
    """
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool))