import os

import torch
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import BasicBlock


class CNN(nn.Module):

    def __init__(self, time_step):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64),
                                    BasicBlock(64, 64),
                                    BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False, padding=0),
            nn.BatchNorm2d(128))),
                                    BasicBlock(128, 128),
                                    BasicBlock(128, 128),
                                    BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=(1, 2), downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=(1, 2), bias=False, padding=0),
            nn.BatchNorm2d(256))),
                                    BasicBlock(256, 256),
                                    BasicBlock(256, 256),
                                    BasicBlock(256, 256),
                                    BasicBlock(256, 256),
                                    BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=(1, 2), downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=(1, 2), bias=False, padding=0),
            nn.BatchNorm2d(512))),
                                    BasicBlock(512, 512),
                                    BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xb):
        out = self.maxpool(self.bn1(self.relu(self.conv1(xb))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.squeeze(dim=3).transpose(1, 2)


class RNN(nn.Module):

    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=0)
        self.last_conv2d = nn.Conv2d(hidden_size * 2, output_size, kernel_size=1)

    def forward(self, xb):
        out, _ = self.lstm(xb)
        out = self.last_conv2d(out.permute(0, 2, 1).unsqueeze(3))
        return out.squeeze(3).permute((2, 0, 1))


class IAMModel(nn.Module):

    def __init__(self, time_step, feature_size, lm,
                 hidden_size, output_size, num_rnn_layers, classes):
        super(IAMModel, self).__init__()
        self.cnn = CNN(time_step=time_step)
        self.rnn = RNN(feature_size=feature_size, hidden_size=hidden_size,
                       output_size=output_size, num_layers=num_rnn_layers)
        self.time_step = time_step

    def forward(self, xb):
        xb = xb.float()
        out = self.cnn(xb)
        out = self.rnn(out)
        return out

    def beam_search_with_lm(self, xb):
        with torch.no_grad():
            out = self.forward(xb)
            # This tensor for each image in the batch contains probabilities of each label for each input feature
            out = out.softmax(2)
            softmax_out = out.permute(1, 0, 2).cpu().numpy()
            char_list = []
            for i in range(softmax_out.shape[0]):
                char_list.append(ctc_beam_search_decoder(probs_seq=softmax_out[i, :],
                                                         alphabet=Alphabet(os.path.abspath("chars.txt")), beam_size=25)[0][1])
        return char_list

    def load_pretrained_resnet(self):
        model_dict = self.state_dict()
        pretrained_dict = load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        pretrained_dict = {f'cnn.{k}': v for k, v in pretrained_dict.items() if f'cnn.{k}' in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)
