import torch
import torch.nn as nn
# from torchsummary import summary


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SpatialMappingModule(nn.Module):
    def __init__(self, dout=0.5):
        super(SpatialMappingModule, self).__init__()
        self.drate = dout
        self.activate = nn.GELU()
        self.cnn1 = nn.Sequential(nn.Conv1d(1, 64, 20, 5), nn.BatchNorm1d(64),
                                  self.activate, nn.Dropout(self.drate),
                                  nn.MaxPool1d(4, 4), nn.Conv1d(64, 128, 4, 1),
                                  nn.BatchNorm1d(128), self.activate,
                                  nn.MaxPool1d(4, 4),
                                  nn.Conv1d(128, 128, 4, 2, padding=1),
                                  nn.BatchNorm1d(128), self.activate)

        self.cnn2 = nn.Sequential(nn.Conv1d(1, 64, 200,
                                            50), nn.BatchNorm1d(64),
                                  self.activate, nn.Dropout(self.drate),
                                  nn.MaxPool1d(4, 1), nn.Conv1d(64, 128, 8, 2),
                                  nn.BatchNorm1d(128), self.activate,
                                  nn.MaxPool1d(4,
                                               1), nn.Conv1d(128, 128, 4, 1),
                                  nn.BatchNorm1d(128), self.activate)
        self.dropout = nn.Dropout(self.drate)
        downsample = nn.Sequential(nn.Conv1d(128, 64, 1, 1, bias=False),
                                   nn.BatchNorm1d(64))
        self.RSEblock = nn.Sequential(
            SEBasicBlock(128, 64, downsample=downsample),
            SEBasicBlock(64, 64),
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_out = self.dropout(x_concat)
        x_out = self.RSEblock(x_out)
        return x_out


class PSENet(nn.Module):
    def __init__(self, num_class=5, width=1024) -> None:
        super(PSENet, self).__init__()
        self.encoder1 = SpatialMappingModule()
        self.encoder2 = SpatialMappingModule()
        classifier_layer_list = [
            torch.nn.Linear(64 * 36, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(width, num_class)
        ]
        self.clf = torch.nn.Sequential(*classifier_layer_list)

    def forward(self, x1, x2):
        out1 = self.encoder1(x1)
        out2 = self.encoder2(x2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        c1 = self.clf(out1)
        c2 = self.clf(out2)
        return c1, c2, out1, out2

    def sum_prd(self, x1, x2):
        out1 = self.encoder1(x1)
        out2 = self.encoder2(x2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return self.clf(out1 + out2)
        # return self.clf(out1 ) + self.clf(out2)

    def predict_x1(self, x):
        out1 = self.encoder1(x)
        out1 = out1.view(out1.shape[0], -1)
        c1 = self.clf(out1)
        return c1

    def predict_x2(self, x):
        out2 = self.encoder2(x)
        out2 = out2.view(out2.shape[0], -1)
        c2 = self.clf(out2)
        return c2
