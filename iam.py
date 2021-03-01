import sys

import Levenshtein as leven
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import rgb2grey
from skimage.transform import rotate
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import matplotlib as mpl
from dataset import IAMData
from model import IAMModel
from colorama import Fore
from tqdm import tqdm

dev = "cuda" if torch.cuda.is_available() else "cpu"

dataset = IAMData(txt_file='./dataset/lines.txt',
                  root_dir='./dataset',
                  output_size=(64, 800),
                  border_pad=(4, 10),
                  random_rotation=2,
                  random_stretch=1.2)

model = IAMModel(channels=3,
                 time_step=96,
                 feature_size=512,
                 hidden_size=512,
                 output_size=len(dataset.char_dict) + 1,
                 num_rnn_layers=4,
                 rnn_dropout=0)
model.load_pretrained_resnet()
model.to(dev)

for p in model.parameters():
    p.requires_grad = True
    model.frozen = [False for i in range(0, len(model.frozen))]


def collate_fn(batch):
    images, words = [b.get('image') for b in batch], [b.get('word') for b in batch]
    images = torch.stack(images, 0)
    lengths = [len(word) for word in words]
    targets = torch.zeros(sum(lengths)).long()
    lengths = torch.tensor(lengths)
    for j, word in enumerate(words):
        start = sum(lengths[:j])
        end = lengths[j]
        targets[start:start + end] = torch.tensor([dataset.char_dict.get(letter) for letter in word]).long()
    return images.to(dev), targets.to(dev), lengths.to(dev)


def fit(model, epochs, train_dl, valid_dl, lr=1e-3, wd=1e-2, betas=(0.9, 0.999)):
    best_leven = 1000
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                     weight_decay=wd, betas=betas)
    len_train = len(train_dl)
    loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)
    for i in range(1, epochs + 1):
        batch_n = 1
        train_loss = 0
        loss = 0
        train_leven = 0
        len_leven = 0
        for xb, yb, lens in tqdm(train_dl,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            model.train()
            opt.zero_grad()
            out = model(xb)
            log_probs = out.log_softmax(2).requires_grad_()
            input_lengths = torch.full((xb.size()[0],), model.time_step, dtype=torch.long)
            loss = loss_func(log_probs, yb, input_lengths, lens)

            with torch.no_grad():
                train_loss += loss

            loss.backward()
            opt.step()

            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.best_path_decode(xb)
                    for j in range(0, len(decoded)):
                        pred_word = decoded[j]
                        actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                        train_leven += leven.distance(''.join(pred_word.astype(str)), ''.join(actual.astype(str)))
                    len_leven += sum(lens).item()

            batch_n += 1

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            leven_dist = 0
            target_lengths = 0
            for xb, yb, lens in tqdm(valid_dl,
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                input_lengths = torch.full((xb.size()[0],), model.time_step, dtype=torch.long)
                valid_loss += loss_func(model(xb).log_softmax(2), yb, input_lengths, lens)
                decoded = model.best_path_decode(xb)
                for j in range(0, len(decoded)):
                    pred_word = decoded[j]
                    actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                    leven_dist += leven.distance(''.join(pred_word.astype(str)), ''.join(actual.astype(str)))
                target_lengths += sum(lens).item()

        print('epoch {}: train loss {} | valid loss {} | \nTRAIN LEVEN {} | VAL LEVEN {}'
              .format(i, train_loss / len(train_dl), valid_loss / len(valid_dl), train_leven / len_leven,
                      leven_dist / target_lengths), end='\n')

        if (leven_dist / target_lengths) < best_leven:
            torch.save(model.state_dict(),
                       f=str((leven_dist / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = leven_dist / target_lengths


batch_size = (120, 240)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
decode_map = {v: k for k, v in dataset.char_dict.items()}
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size[0], sampler=train_sampler, collate_fn=collate_fn)
validation_loader = DataLoader(dataset, batch_size=batch_size[1], sampler=valid_sampler, collate_fn=collate_fn)
fit(model=model, epochs=10, train_dl=train_loader, valid_dl=validation_loader)


def batch_predict(model, valid_dl, up_to):
    xb, yb, lens = iter(valid_dl).next()
    model.eval()
    with torch.no_grad():
        outs = model.best_path_decode(xb)
        for i in range(len(outs)):
            start = sum(lens[:i])
            end = lens[i].item()
            corr = ''.join([decode_map.get(letter.item()) for letter in yb[start:start + end]])
            predicted = ''.join([decode_map.get(letter) for letter in outs[i]])
            img = xb[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            img = rgb2grey(img)
            img = rotate(img, angle=90, clip=False, resize=True)
            f, ax = plt.subplots(1, 1)
            mpl.rcParams["font.size"] = 8
            ax.imshow(img, cmap='gray')
            mpl.rcParams["font.size"] = 14
            plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(corr))
            plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(predicted))
            f.set_size_inches(10, 3)
            print('actual: {}'.format(corr))
            print('predicted:   {}'.format(predicted))
            if i + 1 == up_to:
                break
    plt.show()


batch_predict(model=model, valid_dl=validation_loader, up_to=20)
