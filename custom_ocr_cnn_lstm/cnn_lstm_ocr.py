import torch.nn as nn


class CNN_Encoder(nn.Module):
    def __init__(self, params, leakyRelu=False):
        super(CNN_Encoder, self).__init__()
        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.output_dim = params["output_dim"]
        self.input_planes = params["input_planes"]
        self.planes = params["planes"]
        self.max_width = params["max_width"]  # Add this parameter

        # Define the CNN layers
        # Use 1x1 convolutions for the remaining layers
        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.input_planes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Use 3x3 convolutions for the remaining layers
        self.conv_layer_2 = nn.Conv2d(
            in_channels=self.input_planes,
            out_channels=self.planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        print("cnn input x.shape:", x.shape)
        # Apply the CNN layers
        out = self.conv_layer_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool_1(out)
        out = self.conv_layer_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool_2(out)

        # Get the original height and width
        original_height, original_width = out.shape[2], out.shape[3]
        # Reshape the tensor
        out = out.view(
            out.shape[0], out.shape[1], 1, original_height * original_width
        )
        return out


class LSTM_Decoder(nn.Module):
    def __init__(self, params, leakyRelu=False):
        super(LSTM_Decoder, self).__init__()
        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.output_dim = params["output_dim"] # Alphabet size

        # Define the LSTM layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # input dimensions of x: (sequence_length, batch_size, input_dim)
        output, _ = self.lstm(x)

        # T = length sequence, b = batch size, h = hidden size
        T, b, h = output.size()
        # flattens the batch dimension
        output = self.embedding(output.view(T * b, h))
        # (sequence_length, batch_size, alphabet_size)
        output = output.view(T, b, -1)
        return output

class CNNLSTM_OCR(nn.Module):
    def __init__(self, params):
        super(CNNLSTM_OCR, self).__init__()
        self.cnn_encoder = CNN_Encoder(params)
        params["input_dim"] = 32
        self.lstm_decoder = LSTM_Decoder(params)

    def forward(self, x):
        # Apply the CNN encoder
        out = self.cnn_encoder(x)

        b, c, h, w = out.size()
        print(f"cnn out.size(): {b}, {c}, {h}, {w}")
        assert h == 1, "the height of cnn must be 1"
        # remove all dimensions of size 1
        out = out.squeeze(2)
        out = out.permute(2, 0, 1)  # [w, b, c]
        # Apply the LSTM decoder
        out = self.lstm_decoder(out)
        out = out.transpose(1, 0)  # [b, w, c]
        return out
