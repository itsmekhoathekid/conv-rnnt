import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, bidirectional=True):

        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )

        self.output_proj = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size,
            output_size, bias=True
        )

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)
        
        batch_size = inputs.size(0)
        max_len = inputs.size(1)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)    
        
        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        padded_output = torch.zeros(batch_size, max_len, outputs.size(2))
                
        if inputs.is_cuda: padded_output = padded_output.cuda()

        max_output_size = outputs.size(1)
        padded_output[:, :max_output_size, :] = outputs 
        
        #outputs = self.output_proj(outputs)
        outputs = self.output_proj(padded_output)
        
        return outputs, hidden


import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ProjectedLSTMDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.lstms = nn.ModuleList()
        self.projections = nn.ModuleList()

        for i in range(n_layers):
            input_dim = embedding_size if i == 0 else output_size
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
            proj_dim = 2 * hidden_size if bidirectional else hidden_size
            proj = nn.Sequential(
                nn.Linear(proj_dim, output_size),
                Swish()
            )
            self.lstms.append(lstm)
            self.projections.append(proj)

        # Final projection to match target dimension if needed
        self.output_proj = nn.Linear(output_size, output_size)

    def forward(self, inputs, lengths=None, hidden=None):
        x = self.embedding(inputs)  # [B, T, E]
        B, T, _ = x.shape

        if lengths is not None:
            sorted_lengths, indices = torch.sort(lengths, descending=True)
            x = x[indices]
            restore_indices = torch.argsort(indices)

        for i in range(self.n_layers):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
                self.lstms[i].flatten_parameters()
                packed_out, hidden = self.lstms[i](packed, hidden)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
                x = x[restore_indices]
            else:
                x, hidden = self.lstms[i](x, hidden)

            x = self.projections[i](x)
            x = self.dropout(x)

        out = self.output_proj(x)  # optional linear
        return out


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