import os

import torch
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer
from torch import nn
from torch.utils.model_zoo import load_url
import torchvision.models as models


class CNN(nn.Module):

    def __init__(self, time_step):
        super(CNN, self).__init__()
        resnet18network = models.resnet34()
        resnet18network.load_state_dict(load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth'))
        resnet18network.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.network = torch.nn.Sequential(*(list(resnet18network.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(time_step, 1))

    def forward(self, xb):
        out = self.network(xb)
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

    def __init__(self, time_step, feature_size,
                 hidden_size, output_size, num_rnn_layers):
        super(IAMModel, self).__init__()
        self.cnn = CNN(time_step=time_step)
        self.rnn = RNN(feature_size=feature_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_rnn_layers)
        self.time_step = time_step
        self.alphabet = Alphabet(os.path.abspath("chars.txt"))
        self.scorer = Scorer(alphabet=self.alphabet, scorer_path='iam_uncased.scorer', alpha=0.75, beta=1.85)

    def forward(self, xb):
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
                char_list.append(ctc_beam_search_decoder(probs_seq=softmax_out[i, :], alphabet=self.alphabet, beam_size=25, scorer=self.scorer)[0][1])
        return char_list
