import torch
import torch.nn
from torch.utils.data import random_split, Dataset, DataLoader

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