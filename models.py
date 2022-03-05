"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        char_vectors (torch.Tensor): Character vectors
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
                
        self.emb_char = layers.CharEmbedding(char_vectors=char_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        
        self.emb_word = layers.WordEmbedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=3 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        
        self.self_att = layers.SelfAttention(hidden_size=2 * hidden_size,
                                             drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, wiq, wiqa):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        
        cc_emb = self.emb_char(cc_idxs)
        qc_emb = self.emb_char(qc_idxs)

        cw_emb = self.emb_word(cw_idxs, cc_emb, wiq)     # (batch_size, c_len, hidden_size)
        qw_emb = self.emb_word(qw_idxs, qc_emb, wiqa)     # (batch_size, q_len, hidden_size)

        cw_enc = self.enc(cw_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        qw_enc = self.enc(qw_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(cw_enc, qw_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        att = self.self_att(att)       

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
