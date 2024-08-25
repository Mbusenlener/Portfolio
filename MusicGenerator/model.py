import torch
import torch.nn as nn
from transformers import BertModel
from performer_pytorch import Performer
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)
class MuseNet(nn.Module):
    def __init__(self, music_vocab_size=1025, ff=1024, n_features = 4, d_model=512, nhead=8, num_decoder_layers=6, local_attn_window_size = 150, max_seq_length=4500, dropout=0.1):
        super(MuseNet, self).__init__()
        #Music transformer model. Takes as input tokenized text description labels and discretized latent audio codecs encoded by Encodec
        #Bert encodings are projected with a trainable dense layer that finetunes the outputs for the training task and converts the dimensions
        #Discrete audio codebooks in the shape (Batch size, n_features, seq_length) are embedded and given positional embeddings.
        #Each feature channel is embedded separately, and the final outputs are separately projected back onto the four discrete feature dimensions.
        #Transformer decoder using FAVOR+ attention and rotary positional embeddings applies 
        #causal masked self attention on embedded training musical sequences and cross attends to text encodings.
        #Reversible layers from the Reformer paper and Favor+ attention from the Performer paper are used to handle the computational/memory costs of 4500 token sequences
        self.d_model = d_model
        self.n_features = n_features
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.proj_layer = nn.Linear(768, d_model)
        self.embedding = nn.ModuleList([nn.Embedding(music_vocab_size,d_model) for _ in range(n_features)])
        self.pos_enc = FixedPositionalEmbedding(d_model,max_seq_length)
        self.layer_pos_emb = FixedPositionalEmbedding(64, max_seq_length)
        self.decoder = Performer(
            dim = d_model,
            heads = nhead,
            depth = num_decoder_layers,
            dim_head = 64,
            nb_features=128,
            local_attn_heads = 1,
            local_window_size = local_attn_window_size,
            causal = True,
            ff_mult = 2,
            reversible = True,
            ff_chunks = 2,
            generalized_attention = True,
            use_rezero=True,
            ff_dropout = dropout,
            attn_dropout = dropout,
            cross_attend=True,
            shift_tokens=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.ModuleList([nn.Linear(d_model, music_vocab_size) for _ in range(n_features)])

    def forward(self, src, tgt, src_padding_mask=None):
        src = self.bert_model(src,src_padding_mask).last_hidden_state
        src = self.proj_layer(src)
        tgt = sum([self.embedding[k](tgt[:,k]) for k in range(self.n_features)])
        tgt += self.pos_enc(tgt)
        layer_pos_emb = self.layer_pos_emb(tgt)
        if src_padding_mask is not None:
            src_padding_mask = src_padding_mask.to(torch.bool)
        mask = torch.ones((tgt.size(0),tgt.size(1)),dtype=torch.bool,device=tgt.device)
        memory = src
        output = self.decoder(tgt,context=memory,mask = mask, context_mask=src_padding_mask,pos_emb=layer_pos_emb)
        output = self.norm(output)
        output = torch.stack([self.fc_out[k](output) for k in range(self.n_features)], dim=1)
        return output