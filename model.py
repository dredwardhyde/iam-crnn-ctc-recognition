import numpy as np
import torch
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import BasicBlock

resnet_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'


def downsample(channels, chan_out, stride, pad=0):
    return nn.Sequential(
        nn.Conv2d(channels, chan_out, kernel_size=1, stride=stride, bias=False,
                  padding=pad),
        nn.BatchNorm2d(chan_out)
    )


class CNN(nn.Module):

    def __init__(self, channels, time_step):
        super(CNN, self).__init__()

        self.channels = channels
        if channels == 3:
            self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=2, bias=False)
        else:
            self.chan1_conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(0, 3)])
        self.layer2 = nn.Sequential(*[BasicBlock(64, 128, stride=2, downsample=downsample(64, 128, 2)) if i == 0 else BasicBlock(128, 128) for i in range(0, 4)])
        self.layer3 = nn.Sequential(*[BasicBlock(128, 256, stride=(1, 2), downsample=downsample(128, 256, (1, 2))) if i == 0 else BasicBlock(256, 256) for i in range(0, 6)])
        self.layer4 = nn.Sequential(*[BasicBlock(256, 512, stride=(1, 2), downsample=downsample(256, 512, (1, 2))) if i == 0 else BasicBlock(512, 512) for i in range(0, 3)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xb):
        if self.chan_in == 3:
            out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        else:
            out = self.maxpool(self.bn1(self.relu(self.chan1_conv(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.squeeze(dim=3).transpose(1, 2)


class RNN(nn.Module):

    def __init__(self, feature_size, hidden_size, output_size, num_layers, dropout=0):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.atrous_conv = nn.Conv2d(hidden_size * 2, output_size, kernel_size=1, dilation=1)

    def forward(self, xb):
        out, _ = self.lstm(xb)
        out = self.atrous_conv(out.permute(0, 2, 1).unsqueeze(3))
        return out.squeeze(3).permute((2, 0, 1))


class IAMModel(nn.Module):

    def __init__(self, channels, time_step, feature_size,
                 hidden_size, output_size, num_rnn_layers,
                 rnn_dropout=0):
        super(IAMModel, self).__init__()
        self.cnn = CNN(channels=channels, time_step=time_step)
        self.rnn = RNN(feature_size=feature_size, hidden_size=hidden_size,
                       output_size=output_size, num_layers=num_rnn_layers,
                       dropout=rnn_dropout)
        self.time_step = time_step
        self.to_freeze = []
        self.frozen = []

    def forward(self, xb):
        xb = xb.float()
        out = self.cnn(xb)
        out = self.rnn(out)
        return out

    def best_path_decode(self, xb):
        with torch.no_grad():
            out = self.forward(xb)
            softmax_out = out.softmax(2).argmax(2).permute(1, 0).cpu().numpy()
            char_list = []
            for i in range(0, softmax_out.shape[0]):
                dup_rm = softmax_out[i, :][np.insert(np.diff(softmax_out[i, :]).astype(np.bool), 0, True)]
                dup_rm = dup_rm[dup_rm != 0]
                char_list.append(dup_rm.astype(int))
        return char_list

    def load_pretrained_resnet(self):
        self.to_freeze = []
        self.frozen = []
        model_dict = self.state_dict()
        pretrained_dict = load_url(resnet_url)
        pretrained_dict = {f'cnn.{k}': v for k, v in pretrained_dict.items() if f'cnn.{k}' in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)
        for k in self.state_dict().keys():
            if not 'running' in k and not 'track' in k:
                self.frozen.append(False)
                if k in pretrained_dict.keys():
                    self.to_freeze.append(True)
                else:
                    self.to_freeze.append(False)
