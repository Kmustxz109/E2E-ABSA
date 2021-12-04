import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from torch.nn import CrossEntropyLoss

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.w_2(self.dropout(gelu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super(Encoder, self).__init__()
        # self.word_embed = word_embed
        self.layers = clones(layer, N)
        # print('layer:',layer)
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model,d_model)

    def forward(self, inputs):
        break_probs = []
        x = inputs
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)
        return self.proj(x), break_probs

    def masked_lm_loss(self, out, y):
        fn = CrossEntropyLoss(ignore_index=-1)
        return fn(out.view(-1, out.size()[-1]), y.view(-1))

    def next_sentence_loss(self):
        pass


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, group_prob):
        group_prob, break_prob = self.group_attn(x, group_prob)
        # print('group_prob',group_prob)
        # print('break_prob',break_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob  ############# need

