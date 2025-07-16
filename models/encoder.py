import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=False):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.output_proj = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size,
            output_size, bias=True
        )

        # self.input_bn = nn.BatchNorm1d(input_size)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3  # [B, T, F]

        # print("inputs", inputs.shape)
        B, T, F = inputs.shape
        # inputs = self.input_bn(inputs.view(-1, F)).view(B, T, F)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs_sorted = inputs[indices]


            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs_sorted, sorted_seq_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        else:
            packed_inputs = inputs

        self.lstm.flatten_parameters()

        # print("packed_inputs", packed_inputs.data.shape)

        outputs, hidden = self.lstm(packed_inputs)


        if input_lengths is not None:
            unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)


            _, desorted_indices = torch.sort(indices)
            outputs = unpacked_outputs[desorted_indices]

        else:
            outputs = outputs


        logits = self.output_proj(outputs)


        return logits, hidden

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ProjectedLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.lstms = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(n_layers):
            in_dim = input_size if i == 0 else output_size
            lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
            out_dim = 2 * hidden_size if bidirectional else hidden_size
            proj = nn.Sequential(
                nn.Linear(out_dim, output_size),
                Swish()
            )
            self.lstms.append(lstm)
            self.projections.append(proj)
        # self.output_proj = nn.Linear(
        #     2 * hidden_size if bidirectional else hidden_size,
        #     output_size, bias=True
        # )

    def forward(self, x, lengths=None):
        B, T, F = x.shape
        if lengths is not None:
            sorted_lengths, indices = torch.sort(lengths, descending=True)
            x = x[indices]
            restore_indices = torch.argsort(indices)

        for i in range(self.n_layers):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
                self.lstms[i].flatten_parameters()
                packed_out, _ = self.lstms[i](packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
                x = x[restore_indices]
            else:
                x, _ = self.lstms[i](x)

            x = self.projections[i](x)
            x = self.drop(x)
        # x = self.output_proj(x) 
        return x  # shape: [B, T, output_size]



def build_encoder(config):
    if config["enc"]["type"] == 'lstm':
        return ProjectedLSTMEncoder(
            input_size=config["feature_dim"],
            hidden_size=config["enc"]["hidden_size"],
            output_size=config["enc"]["output_size"],
            n_layers=config["enc"]["n_layers"],
            dropout=config["dropout"],
            bidirectional=config["enc"]["bidirectional"]
        )
    else:
        raise NotImplementedError("Encoder type not implemented.")