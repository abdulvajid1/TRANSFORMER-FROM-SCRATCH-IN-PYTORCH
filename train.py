import torch
import torch.nn
from torch.utils.data import random_split, Dataset, DataLoader

from dataset import BilingualDataset
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models  import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config(['tokenizer_file'].format(lang)))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens =["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequeny=2)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_ds(config):
    ds_raw = load_dataset('opus_book', f"{config['lang_src']}--{config['lang_tgt']}", split=True)

    # Build tokenizers 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # keep 90% percentage for training and 10% for validation 
    train_ds_size = 0.90 * len(ds_raw)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_src = max(max_len_src, len(src_ids))
        
    print(f'Max length of source language : {max_len_src}')
    print(f'Max length of target langauge : {max_len_tgt}')

    train_dataloader = DataLoader(train_ds,  batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds,  batch_size=1, shuffle=True)

def get_model(config, vocab_src_size, vocab_tgt_size):
    model = build_transformer(vocab_src_size, vocab_tgt_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')