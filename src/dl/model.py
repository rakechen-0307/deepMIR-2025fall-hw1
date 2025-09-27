import torch

class Conv_2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = torch.nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class ShortChunkCNN(torch.nn.Module):
    """
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    """
    def __init__(
        self, used_spec, sr=16000, n_class=20,
        mel_n_channels=64, cqt_n_channels=64, 
        dropout=0.5
    ):
        super(ShortChunkCNN, self).__init__()

        self.sr = sr
        self.n_class = n_class
        self.used_spec = used_spec

        self.spec_bn = torch.nn.BatchNorm2d(1)

        # Mel Spectrogram CNN
        self.mel_layer1 = Conv_2d(1, mel_n_channels, pooling=2)
        self.mel_layer2 = Conv_2d(mel_n_channels, mel_n_channels, pooling=2)
        self.mel_layer3 = Conv_2d(mel_n_channels, mel_n_channels, pooling=2)
        self.mel_layer4 = Conv_2d(mel_n_channels, mel_n_channels*2, pooling=2)
        self.mel_layer5 = Conv_2d(mel_n_channels*2, mel_n_channels*2, pooling=2)
        self.mel_layer6 = Conv_2d(mel_n_channels*2, mel_n_channels*2, pooling=2)
        self.mel_layer7 = Conv_2d(mel_n_channels*2, mel_n_channels*4, pooling=2)

        # CQT CNN
        self.cqt_layer1 = Conv_2d(1, cqt_n_channels, pooling=1)
        self.cqt_layer2 = Conv_2d(cqt_n_channels, cqt_n_channels, pooling=2)
        self.cqt_layer3 = Conv_2d(cqt_n_channels, cqt_n_channels, pooling=2)
        self.cqt_layer4 = Conv_2d(cqt_n_channels, cqt_n_channels*2, pooling=2)
        self.cqt_layer5 = Conv_2d(cqt_n_channels*2, cqt_n_channels*2, pooling=2)
        self.cqt_layer6 = Conv_2d(cqt_n_channels*2, cqt_n_channels*2, pooling=2)
        self.cqt_layer7 = Conv_2d(cqt_n_channels*2, cqt_n_channels*4, pooling=2)

        # Dense
        if "mel" in self.used_spec and "cqt" in self.used_spec:
            n_channels = mel_n_channels*4 + cqt_n_channels*4
        elif "mel" in self.used_spec:
            n_channels = mel_n_channels*4
        elif "cqt" in self.used_spec:
            n_channels = cqt_n_channels*4
        else:
            raise ValueError("At least one of 'mel' or 'cqt' must be used in used_spec.")

        self.dense1 = torch.nn.Linear(n_channels, n_channels // 4)
        self.bn = torch.nn.BatchNorm1d(n_channels // 4)
        self.dense2 = torch.nn.Linear(n_channels // 4, n_class)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, mel, cqt):
        # Spectrogram
        if "mel" in self.used_spec:
            mel = self.spec_bn(mel)
            mel = self.mel_layer1(mel)
            mel = self.mel_layer2(mel)
            mel = self.mel_layer3(mel)
            mel = self.mel_layer4(mel)
            mel = self.mel_layer5(mel)
            mel = self.mel_layer6(mel)
            mel = self.mel_layer7(mel)
            mel = mel.squeeze(2)  # [batch, channels, time]

            if mel.size(-1) != 1:
                mel = torch.nn.MaxPool1d(mel.size(-1))(mel)
            mel = mel.squeeze(2)

        if "cqt" in self.used_spec:
            cqt = self.spec_bn(cqt)
            cqt = self.cqt_layer1(cqt)
            cqt = self.cqt_layer2(cqt)
            cqt = self.cqt_layer3(cqt)
            cqt = self.cqt_layer4(cqt)
            cqt = self.cqt_layer5(cqt)
            cqt = self.cqt_layer6(cqt)
            cqt = self.cqt_layer7(cqt)
            cqt = cqt.squeeze(2)  # [batch, channels, time]

            if cqt.size(-1) != 1:
                cqt = torch.nn.MaxPool1d(cqt.size(-1))(cqt)
            cqt = cqt.squeeze(2)

        if "mel" in self.used_spec and "cqt" in self.used_spec:
            x = torch.cat([mel, cqt], dim=1)
        elif "mel" in self.used_spec:
            x = mel
        elif "cqt" in self.used_spec:
            x = cqt
        else:
            raise ValueError("At least one of 'mel' or 'cqt' must be used in used_spec.")

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x