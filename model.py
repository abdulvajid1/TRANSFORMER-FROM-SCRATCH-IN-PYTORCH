import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pe = torch.zeros(seq_len,d_model)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000)/d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Positonal encodings
        pe[:,::2] = torch.sin(positions * div_term)
        pe[:,1::2] = torch.cos(positions * div_term)
        pe.unsqueeze_(0)
        # make it non-trainable, does not consider as parameter
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.size(1),:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Scaling
        self.bias = nn.Parameter(torch.zeros(1)) # Adding
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, dim_ff:int,dropout:float):
        super().__init__()
        self.ff1 = nn.Linear(d_model,dim_ff)
        self.ff2 = nn.Linear(dim_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.ff1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x

# Attention
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, heads:int=8):
        super().__init__()
        # Initialize Q, K, V metrix
        self.query_weights = nn.Linear(d_model, d_model)
        self.key_weights = nn.Linear(d_model, d_model)
        self.value_weights = nn.Linear(d_model, d_model)
        self.output_weights = nn.Linear(d_model, d_model)
        self.heads = heads
        assert d_model % heads == 0, "d_model should be dividable by heads"
        self.d_k = d_model // self.heads
        
    def forward(self,query, key, value, mask):
        queries = self.query_weights(query) # (batch,seq,d_model)
        keys = self.key_weights(key)
        values = self.value_weights(value)

        # Splitting heads,(batch,seq,d_model) -> (batchsize,seq,heads,d_k) # (batch, head, seq, d_k)
        queries = queries.view(queries.size(0), queries.size(1), self.heads, self.d_k).transpose(1,2)  
        keys = keys.view(keys.size(0), keys.size(1), self.heads, self.d_k).transpose(1,2)
        values = values.view(values.size(0), values.size(1), self.heads, self.d_k).transpose(1,2)

        # Attention
        context_vectors_heads ,attention_scores = MultiHeadAttention.attention_score(queries, keys, values, mask)
        context_vectors = context_vectors_heads.transpose(1, 2).contiguous().view(context_vectors_heads.size(0), context_vectors_heads.size(2), self.heads*self.d_k) #batch,h,seq,dk -> batch,seq,h,dk -> batch,seq,d_model
        return self.output_weights(context_vectors)

    @staticmethod
    def attention_score(queries, keys, values, mask):
        d_k = queries.size(-1)
        attention_score = (queries @ keys.transpose(-1,-2)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0, -1e9) # masking with very small value instead of '-inf'
        
        # Context vectors
        attention_score_softmax = attention_score.softmax(dim=-1)
        context_vectores_heads = attention_score_softmax @ values
        return context_vectores_heads, attention_score  

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        x_norm = self.norm(x)
        sublayer_out = sublayer(x_norm)
        x = x + sublayer_out
        return self.dropout(x) 

        
class EncoderBlock(nn.Module):
    
    def __init__(self, multihead_attention:MultiHeadAttention, feedforward:FeedForwardBlock, dropout:float):
        super().__init__()
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.multihead_attention = multihead_attention
        self.feedforward = feedforward
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x, src_mask):
        x = self.residual_connections[0](x, lambda x :self.multihead_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward)
        return self.dropout(x)
    
class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention:MultiHeadAttention, feedforward:FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output,src_mask, tar_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tar_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward)
        return x   

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.decoder_layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask ,tar_mask):
        for decoder_block in self.decoder_layers:
            x = decoder_block(x, encoder_output, src_mask, tar_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.projection(x)
    

class Transformer(nn.Module):
    def __init__(self,
                 encoder:Encoder, 
                 decoder:Decoder, 
                 src_embed:InputEmbedding, 
                 tgt_embed:InputEmbedding, 
                 src_pos:PositionalEncoding, 
                 tgt_pos:PositionalEncoding, 
                 projection_layer:ProjectionLayer):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, x, encoder_output, encoder_mask, tar_mask):
        x = self.tgt_embed(x)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, encoder_mask, tar_mask)
    
    def projection(self, x):
        return self.projection_layer(x)
    

def build_transformer( 
                      src_vocab_size:int, 
                      tgt_vocab_size:int, 
                      src_seq_len:int, 
                      tgt_seq_len:int,
                      d_model:int=512,
                      dropout:float=0.1,
                      heads=8,
                      N:int=6,
                      d_ff:int=2048):

    src_embed = InputEmbedding(d_model,src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Ecnoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_attention = MultiHeadAttention(d_model, heads)
        feedforward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_attention, feedforward, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, heads)
        decoder_cross_attention = MultiHeadAttention(d_model,heads)
        feedforward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feedforward, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer

 

