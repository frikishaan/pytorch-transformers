from datasets import load_dataset
import sentencepiece as spm
import json

# load config
with open("config.json", "r") as f:
    config = json.load(f)

ds = load_dataset("cfilt/iitb-english-hindi", split="train")

# Generate corpus from dataset
with open("hi_corpus.txt", "w", encoding="utf-8") as f_hi, \
     open("en_corpus.txt", "w", encoding="utf-8") as f_en:
    for item in ds["translation"]:
        f_hi.write(item["hi"].strip() + "\n")
        f_en.write(item["en"].strip() + "\n")

# Train tokenizer
spm.SentencePieceTrainer.train(
    input="hi_corpus.txt",
    model_prefix=config["tokenizer"]["hi_path"],
    model_type="unigram",
    vocab_size=config["vocab"]["size"],
    character_coverage=1.0,
    unk_id=config["vocab"]["unk_id"],
    pad_id=config["vocab"]["pad_id"],
    bos_id=config["vocab"]["bos_id"],
    eos_id=config["vocab"]["eos_id"],
    unk_piece=config["vocab"]["unk_token"],
    pad_piece=config["vocab"]["pad_token"],
    bos_piece=config["vocab"]["bos_token"],
    eos_piece=config["vocab"]["eos_token"],
)

spm.SentencePieceTrainer.train(
    input="en_corpus.txt",
    model_prefix=config["tokenizer"]["en_path"],
    model_type="unigram",
    vocab_size=config["vocab"]["size"],
    character_coverage=1.0,
    unk_id=config["vocab"]["unk_id"],
    pad_id=config["vocab"]["pad_id"],
    bos_id=config["vocab"]["bos_id"],
    eos_id=config["vocab"]["eos_id"],
    unk_piece=config["vocab"]["unk_token"],
    pad_piece=config["vocab"]["pad_token"],
    bos_piece=config["vocab"]["bos_token"],
    eos_piece=config["vocab"]["eos_token"],
)

# Load tokenizer
hi_sp = spm.SentencePieceProcessor()
hi_sp.load(config["tokenizer"]["hi_path"] + ".model")

en_sp = spm.SentencePieceProcessor()
en_sp.load(config["tokenizer"]["en_path"] + ".model")

print(f"Hindi vocab size : {hi_sp.get_piece_size()}")
print(f"English vocab size : {en_sp.get_piece_size()}")

def get_coverage(sp, corpus):
    total_tokens, unk_tokens = 0, 0
    with open(corpus, 'r', encoding='utf-8') as f:
        for line in f:
            ids = sp.encode(line.strip(), out_type=int)
            total_tokens += len(ids)
            unk_tokens += sum(1 for i in ids if i == sp.unk_id())

    coverage = 100 * (1 - unk_tokens / total_tokens)
    return coverage

print(f"Hindi coverage: {get_coverage(hi_sp, 'hi_corpus.txt'):.2f}%")
print(f"English coverage: {get_coverage(en_sp, 'en_corpus.txt'):.2f}%")

# Delete corpus files
import os
print(f"Deleting corpus files...")
os.remove("hi_corpus.txt")
os.remove("en_corpus.txt")