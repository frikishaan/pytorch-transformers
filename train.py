from dataclasses import dataclass
import glob
from typing import Tuple
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import nn
from torch.amp import autocast, GradScaler
import json
import argparse

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
from dataset import TranslationDataset, causal_mask
from model import Transformer, build_transformer
from datetime import datetime

from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm

torch.manual_seed(24)
warnings.filterwarnings("ignore")

writer = SummaryWriter("transformers")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detected - {device}")

@dataclass(frozen=True)
class Config:
    vocab_size: int
    d_model: int
    seq_len: int
    dropout: float
    n_heads: int
    n_layers: int
    d_ff: int
    model_basename: str
    en_tokenizer_path: str
    hi_tokenizer_path: str
    batch_size: int
    grad_acc_steps: int
    learning_rate: float
    warmup_steps: int
    
def load_config(json_path: str) -> Config:
    """Loads configuration from a JSON file and returns a Config object."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    return Config(**config_dict)

def get_tokenizer(config: Config) -> Tuple[SentencePieceProcessor, SentencePieceProcessor]:
    src_sp = SentencePieceProcessor()
    tgt_sp = SentencePieceProcessor()
    src_sp.load(config.en_tokenizer_path + ".model")
    tgt_sp.load(config.hi_tokenizer_path + ".model")
    return src_sp, tgt_sp

def get_ds(config: Config, src_sp: SentencePieceProcessor, tgt_sp: SentencePieceProcessor):
    
    train_dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
    train_dataset = train_dataset.map(add_token_ids, fn_kwargs={"config": config, "src_sp": src_sp, "tgt_sp": tgt_sp})

    val_dataset = load_dataset("cfilt/iitb-english-hindi", split="validation")
    val_dataset = val_dataset.map(add_token_ids, fn_kwargs={"config": config, "src_sp": src_sp, "tgt_sp": tgt_sp})

    # filter dataset
    train_dataset = train_dataset.filter(filter_ds, fn_kwargs={"config": config})
    val_dataset = val_dataset.filter(filter_ds, fn_kwargs={"config": config})

    return train_dataset, val_dataset

def filter_ds(row, config: Config):
    """
    Filter out examples with empty and too long rows 
    """
    try:
        return len(row["src_ids"]) > 0 and len(row["tgt_ids"]) > 0 and len(row["src_ids"]) <= config.seq_len - 2 and len(row["tgt_ids"]) <= config.seq_len - 2
    except:
        return False

def add_token_ids(row, config: Config, src_sp: SentencePieceProcessor, tgt_sp: SentencePieceProcessor):
    row["src_ids"] = src_sp.encode(row["translation"]["en"].strip(), out_type=int)
    row["tgt_ids"] = tgt_sp.encode(row["translation"]["hi"].strip(), out_type=int)
    return row

def run_validation(model: Transformer, val_loader: DataLoader, tgt_sp: SentencePieceProcessor, config: Config, epoch: int) -> None:
    """
    Run validation with Teacher forcing
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tgt_sp.pad_id(),
        label_smoothing=0.1,
    ).to(device)

    total_loss = 0.0

    batch_iterator = tqdm(val_loader, desc=f"Running validation for epoch: {epoch:02d}")
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, proj_output.shape[-1]),
                label.view(-1),
            )

            total_loss += loss.item()
            batch_iterator.set_postfix({ "Loss": f"{loss.item():6.3f}"})

    writer.add_scalar("Validation loss per epoch", total_loss / len(val_loader), epoch)
    writer.flush()

    # BLEU and other metrics

def calculate_metrics(model: Transformer, val_loader: DataLoader, tgt_sp: SentencePieceProcessor, config: Config, epoch: int):
    model.eval()
    batch_iterator = tqdm(val_loader, desc=f"Calculating metrics for epoch {epoch}")

    expected = []
    predicted = []
    
    with torch.no_grad():
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)
            
            decoder_input = torch.full((config.batch_size, 1), tgt_sp.bos_id()).type_as(encoder_input).to(device)
            finished = torch.zeros(config.batch_size, dtype=bool)

            # encode once
            encoder_output = model.encode(encoder_input, encoder_mask)

            for t in range(config.seq_len - 1):
                decoder_mask = causal_mask(decoder_input.size(1)).to(device)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                logits = model.project(decoder_output[:, -1]) # (batch, vocab)

                next_tok = torch.argmax(logits, dim=-1) # (batch,)

                # force EOS for finished sequences
                next_tok[finished] = tgt_sp.eos_id()

                finished |= (next_tok == tgt_sp.eos_id())

                decoder_input = torch.cat([
                    decoder_input,
                    next_tok.unsqueeze(1)
                ], dim=1)

                if finished.all():
                    break

            predicted += tgt_sp._DecodeIdsBatch(decoder_input.squeeze(0).detach().cpu().numpy().tolist(), 1)
            expected += tgt_sp._DecodeIdsBatch(label.squeeze(0).detach().cpu().numpy().tolist(), 1)

    bleu = BLEUScore()
    score = bleu(predicted, [[e] for e in expected])
    print(f"BLEU: {score.item():6.3f}")
    writer.add_scalar("BLEU per epoch", score.item(), epoch)
    writer.flush()

    wer = WordErrorRate()
    w_score = wer(predicted, expected)
    print(f"WordErrorRate: {w_score.item():6.3f}")
    writer.add_scalar("WordErrorRate per epoch", w_score.item(), epoch)
    writer.flush()

    cer = CharErrorRate()
    c_score = cer(predicted, expected)
    print(f"CharacterErrorRate: {c_score.item():6.3f}")
    writer.add_scalar("CharacterErrorRate per epoch", c_score.item(), epoch)
    writer.flush()

def get_model_path(config: Config) -> str:    
    # Get latest model path
    model_path = glob.glob(config.model_basename + "_epoch_*.pt")
    if len(model_path) == 0:
        return None
    return sorted(model_path)[-1]

def make_noam_scheduler(optimizer, d_model: int, warmup_steps: int):
    # lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    def lr_lambda(step: int):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    args = parser.parse_args()
    epochs = args.epochs

    config = load_config("config.json")

    src_sp, tgt_sp = get_tokenizer(config)

    src_vocab_size = src_sp.get_piece_size()
    tgt_vocab_size = tgt_sp.get_piece_size()

    src_pad_id = src_sp.pad_id()
    tgt_pad_id = tgt_sp.pad_id()
    src_unk_id = src_sp.unk_id()
    tgt_unk_id = tgt_sp.unk_id()
    src_bos_id = src_sp.bos_id()
    tgt_bos_id = tgt_sp.bos_id()
    src_eos_id = src_sp.eos_id()
    tgt_eos_id = tgt_sp.eos_id()
    
    train_ds, val_ds = get_ds(config, src_sp, tgt_sp)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("No valid training or validation samples found after filtering")

    train_loader = DataLoader(
        TranslationDataset(train_ds, src_sp, tgt_sp, config.seq_len), 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True,
        # pin_memory=True,
        # num_workers=4,
        # persistent_workers=True
    )

    val_loader = DataLoader(
        TranslationDataset(val_ds, src_sp, tgt_sp, config.seq_len), 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True,
        # pin_memory=True,
        # num_workers=4,
        # persistent_workers=True
    )

    # Create transformer model
    model = build_transformer(
        src_sp.vocab_size(),
        tgt_sp.vocab_size(),
        config.seq_len,
        config.seq_len,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.dropout,
        config.d_ff
    ).to(device)

    # define loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = make_noam_scheduler(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_sp.pad_id(), label_smoothing=0.1).to(device)

    global_step = 0
    initial_epoch = 0
    train_loss = 0

    # load from file if available
    model_path = get_model_path(config)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initial_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        print(f"Loaded model from {model_path}")
    except (FileNotFoundError, RuntimeError, Exception) as e:
        print(f"Initializing new model")

    # calculate_metrics(model, val_loader, tgt_sp, config, 2)
    start_time = datetime.now()

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(initial_epoch + 1, initial_epoch + epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                # Predict
                encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
                proj_output = model.project(decoder_output) # (batch, seq_len, vocab_size)

                # Compute loss
                loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))

            batch_iterator.set_postfix({ "Loss": f"{loss.item():6.3f}"})
            train_loss += loss.item()

            # log the loss
            writer.add_scalar('Train loss per step', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            scaler.scale(loss).backward()

            # Update the weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1

        # run validation
        run_validation(model, val_loader, tgt_sp, config, epoch)

        calculate_metrics(model, val_loader, tgt_sp, config, epoch)

        # average loss for epoch
        train_loss = train_loss / len(train_loader)

        writer.add_scalar('Train loss per epoch', train_loss, epoch)
        writer.flush()

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': global_step,
            'epoch': epoch
        }, config.model_basename + f"_epoch_{epoch}.pt")
    
    end_time = datetime.now()
    print(f"Finished training in {(end_time - start_time).total_seconds() // 60} minutes")
