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
        self.conv_layer_1 = nn.Conv2d(self.input_planes, 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Use 3x3 convolutions for the remaining layers
        self.conv_layer_2 = nn.Conv2d(self.input_planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the CNN layers
        out = self.conv_layer_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool_1(out)
        out = self.conv_layer_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool_2(out)
        return out

class LSTM_Decoder(nn.Module):

    def __init__(self, params, leakyRelu=False):
        super(LSTM_Decoder, self).__init__()
        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.output_dim = params["output_dim"]

        # Define the LSTM layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        # Apply the LSTM layers
        output = self.lstm(x)
        output = output.transpose(1,0) #Tbh to bth
        return output

class CNNLSTM_OCR(nn.Module):
    def __init__(self, params):
        super(CNNLSTM_OCR, self).__init__()
        self.cnn_encoder = CNN_Encoder(params)
        self.lstm_decoder = LSTM_Decoder(params)
    
    def forward(self, x):
        # Apply the CNN encoder
        out = self.cnn_encoder(x)
        # Apply the LSTM decoder
        out = self.lstm_decoder(out)
        return out
