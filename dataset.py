import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        enc_input_token = self.tokenizer_src.encode(src_text)
        dec_input_token = self.tokenizer_tgt.encode(tgt_text)

        enc_num_paddings  = self.seq_len - len(enc_input_token)  - 2 # We will add 2 more token at the end (sos,eos)
        dec_num_paddings = self.seq_len - len(dec_num_paddings) - 1 # Training time we only add sos token and in the label we add eos 

        if enc_num_paddings < 0 or dec_num_paddings < 0 :
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_paddings, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_paddings, dtype=torch.int64)
        ])

        decoder_label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_paddings, dtype=torch.int64)

        ])
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert decoder_label.size(0) == self.seq_len 

        return {
            'encoder_input' : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input != self.pad_token & casual_mask(decoder_input.size(0))),
            'label' : decoder_label
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1)
    return mask == 0


        
