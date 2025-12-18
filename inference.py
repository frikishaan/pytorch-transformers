import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from sentencepiece import SentencePieceProcessor

from dataset import causal_mask
from model import Transformer, build_transformer


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data)


def load_tokenizers(config: Config) -> Tuple[SentencePieceProcessor, SentencePieceProcessor]:
    src_sp = SentencePieceProcessor()
    tgt_sp = SentencePieceProcessor()
    src_sp.load(config.en_tokenizer_path + ".model")
    tgt_sp.load(config.hi_tokenizer_path + ".model")
    return src_sp, tgt_sp


def select_checkpoint(config: Config, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        explicit = Path(explicit_path)
        return explicit if explicit.is_absolute() else (BASE_DIR / explicit)

    base = Path(config.model_basename)
    if not base.is_absolute():
        base = BASE_DIR / base
    pattern = str(base.parent / f"{base.name}_epoch_*.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found matching pattern: {pattern}")
    return Path(candidates[-1])


def load_model(
    config: Config,
    src_sp: SentencePieceProcessor,
    tgt_sp: SentencePieceProcessor,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> Transformer:
    model = build_transformer(
        src_sp.vocab_size(),
        tgt_sp.vocab_size(),
        config.seq_len,
        config.seq_len,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.dropout,
        config.d_ff,
    ).to(device)

    checkpoint_file = select_checkpoint(config, checkpoint_path)
    state = torch.load(checkpoint_file, map_location=device)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_encoder_inputs(
    text: str,
    src_sp: SentencePieceProcessor,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = src_sp.encode(text.strip(), out_type=int)
    truncated = False
    max_tokens = config.seq_len - 1  # reserve space for EOS
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        truncated = True

    sequence = tokens + [src_sp.eos_id()]
    padding_needed = config.seq_len - len(sequence)
    sequence.extend([src_sp.pad_id()] * padding_needed)

    if truncated:
        print("Warning: input truncated to fit model sequence length.", file=sys.stderr)

    encoder_input = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    encoder_mask = (encoder_input != src_sp.pad_id()).unsqueeze(1).unsqueeze(2)
    return encoder_input, encoder_mask


def greedy_search(
    model: Transformer,
    encoder_input: torch.Tensor,
    encoder_mask: torch.Tensor,
    tgt_sp: SentencePieceProcessor,
    max_len: int,
) -> torch.Tensor:
    device = encoder_input.device
    model.eval()

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_input = torch.tensor([[tgt_sp.bos_id()]], dtype=torch.long, device=device)
        finished = torch.zeros(1, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            decoder_mask = causal_mask(decoder_input.size(1)).to(device)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(decoder_output)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.masked_fill(finished, tgt_sp.eos_id())

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            finished |= next_token == tgt_sp.eos_id()

            if finished.all():
                break

    return decoder_input.squeeze(0)


def beam_search(
    model: Transformer,
    encoder_input: torch.Tensor,
    encoder_mask: torch.Tensor,
    tgt_sp: SentencePieceProcessor,
    max_len: int,
    beam_size: int,
) -> torch.Tensor:
    device = encoder_input.device
    beam_size = max(1, beam_size)

    model.eval()

    bos_id = tgt_sp.bos_id()
    eos_id = tgt_sp.eos_id()

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)

        initial_seq = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        beams = [(initial_seq, 0.0, False)]  # (tokens, log_prob, finished)

        for _ in range(max_len - 1):
            candidates = []

            # Carry over already finished beams unchanged.
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))

            active = [(idx, b) for idx, b in enumerate(beams) if not b[2]]
            if not active:
                beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
                break

            decoder_inputs = torch.cat([b[0] for _, b in active], dim=0)
            step_len = decoder_inputs.size(1)

            base_decoder_mask = causal_mask(step_len).to(device)
            decoder_mask = base_decoder_mask.expand(decoder_inputs.size(0), -1, -1).unsqueeze(1)
            expanded_encoder_output = encoder_output.expand(decoder_inputs.size(0), -1, -1)
            expanded_encoder_mask = encoder_mask.expand(decoder_inputs.size(0), *encoder_mask.shape[1:])

            decoder_output = model.decode(
                expanded_encoder_output,
                expanded_encoder_mask,
                decoder_inputs,
                decoder_mask,
            )
            logits = model.project(decoder_output)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

            for (idx, (seq, score, _)), row_log_probs, row_indices in zip(active, topk_log_probs, topk_indices):
                for token_log_prob, token_id in zip(row_log_probs, row_indices):
                    next_token = token_id.view(1, 1)
                    new_seq = torch.cat([seq, next_token.to(device)], dim=1)
                    new_score = score + float(token_log_prob.item())
                    finished = token_id.item() == eos_id
                    candidates.append((new_seq, new_score, finished))

            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]

            if all(finished for _, _, finished in beams):
                break

        finished_beams = [b for b in beams if b[2]]
        best_beam = max(finished_beams, key=lambda item: item[1]) if finished_beams else max(beams, key=lambda item: item[1])

    return best_beam[0].squeeze(0)


def tokens_to_text(tokens: torch.Tensor, tgt_sp: SentencePieceProcessor) -> str:
    token_list = tokens.tolist()
    if token_list and token_list[0] == tgt_sp.bos_id():
        token_list = token_list[1:]

    cleaned = []
    for token in token_list:
        if token == tgt_sp.eos_id():
            break
        if token == tgt_sp.pad_id():
            continue
        cleaned.append(token)

    return tgt_sp.decode(cleaned)


BASE_DIR = Path(__file__).resolve().parent
CONFIG = load_config(str(BASE_DIR / "config.json"))
DEVICE = _resolve_device()
SRC_SP, TGT_SP = load_tokenizers(CONFIG)
MODEL = load_model(CONFIG, SRC_SP, TGT_SP, DEVICE)


def generate_response(text: str, max_new_tokens: Optional[int] = None) -> str:
    max_len = max(2, max_new_tokens or CONFIG.seq_len)
    encoder_input, encoder_mask = prepare_encoder_inputs(text, SRC_SP, CONFIG, DEVICE)
    generated_tokens_b = beam_search(MODEL, encoder_input, encoder_mask, TGT_SP, max_len, 3)
    generated_tokens = greedy_search(MODEL, encoder_input, encoder_mask, TGT_SP, max_len)
    return tokens_to_text(generated_tokens_b, TGT_SP), tokens_to_text(generated_tokens, TGT_SP)


def chat_loop() -> None:
    while True:
        try:
            user_text = input("Input: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            break

        beam_resp, greedy_resp = generate_response(user_text)
        print(f"Beam: {beam_resp}")
        print(f"Greedy: {greedy_resp}")


def main() -> None:
    chat_loop()


if __name__ == "__main__":
    main()
