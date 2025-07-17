import torch
import torch.nn as nn
from .encoder import BaseLSTMLayer

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ProjectedLSTMDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            input_dim = embedding_size if i == 0 else output_size
            layer = BaseLSTMLayer(
                input_size=input_dim,
                hidden_size=hidden_size,
                output_size=output_size,
                n_layers=1,  # Each layer is a single LSTM layer
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.layers.append(layer)

    def forward(self, inputs, lengths=None, hidden=None):
        x = self.embedding(inputs)  # [B, T, E]

        for layer in self.layers:
            x, hidden = layer(x, lengths)
        return x, hidden


def build_decoder(config):
    if config["dec"]["type"] == 'lstm':
        return ProjectedLSTMDecoder(
            embedding_size=config["dec"]["embedding_size"],
            hidden_size=config["dec"]["hidden_size"],
            vocab_size=config["vocab_size"],
            output_size=config["dec"]["output_size"],
            n_layers=config["dec"]["n_layers"],
            dropout=config["dropout"],
        )
    else:
        raise NotImplementedError("Decoder type not implemented.")