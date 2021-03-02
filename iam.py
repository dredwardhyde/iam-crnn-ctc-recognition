import sys

import Levenshtein as leven
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore
from skimage.color import rgb2grey
from skimage.transform import rotate
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm

from dataset import IAMData
from model import IAMModel

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================= PREPARING DATASET ======================================================
dataset = IAMData(txt_file='./dataset/lines.txt',
                  root_dir='./dataset',
                  output_size=(64, 800),
                  border_pad=(4, 10),
                  random_rotation=2,
                  random_stretch=1.2)


def collate(batch):
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


# ================================================= MODEL ==============================================================
model = IAMModel(time_step=96,
                 feature_size=512,
                 hidden_size=512,
                 output_size=len(dataset.char_dict) + 1,
                 num_rnn_layers=4,
                 rnn_dropout=0)
model.load_pretrained_resnet()
model.to(dev)
# Unfreeze ResNet34 layers
for p in model.parameters():
    p.requires_grad = True
    model.frozen = [False for i in range(0, len(model.frozen))]


# ================================================ TRAINING MODEL ======================================================
def fit(model, epochs, train_data_loader, valid_data_loader, lr=1e-3, wd=1e-2, betas=(0.9, 0.999)):
    best_leven = 1000
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                     weight_decay=wd, betas=betas)
    len_train = len(train_data_loader)
    loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)
    for i in range(1, epochs + 1):
        # ============================================ TRAINING ========================================================
        batch_n = 1
        loss = 0
        train_levenshtein = 0
        len_levenshtein = 0
        for xb, yb, lens in tqdm(train_data_loader,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            model.train()
            opt.zero_grad()
            input_lengths = torch.full((xb.size()[0],), model.time_step, dtype=torch.long)
            loss = loss_func(model(xb), yb, input_lengths, lens)
            loss.backward()
            opt.step()
            # ================================== TRAINING LEVENSHTEIN DISTANCE =========================================
            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.eager_decode(xb)
                    for j in range(0, len(decoded)):
                        actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                        train_levenshtein += leven.distance(''.join(decoded[j].astype(str)), ''.join(actual.astype(str)))
                    len_levenshtein += sum(lens).item()

            batch_n += 1

        # ============================================ VALIDATION ======================================================
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            val_levenshtein = 0
            target_lengths = 0
            for xb, yb, lens in tqdm(valid_data_loader,
                                     position=0, leave=True,
                                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                input_lengths = torch.full((xb.size()[0],), model.time_step, dtype=torch.long)
                valid_loss += loss_func(model(xb), yb, input_lengths, lens)
                decoded = model.eager_decode(xb)
                for j in range(0, len(decoded)):
                    actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                    val_levenshtein += leven.distance(''.join(decoded[j].astype(str)), ''.join(actual.astype(str)))
                target_lengths += sum(lens).item()

        print('epoch {}: Train Levenshtein {} | Validation Levenshtein {}'
              .format(i, train_levenshtein / len_levenshtein, val_levenshtein / target_lengths), end='\n')
        # ============================================ SAVE MODEL ======================================================
        if (val_levenshtein / target_lengths) < best_leven:
            torch.save(model.state_dict(), f=str((val_levenshtein / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = val_levenshtein / target_lengths


train_batch_size = 120
validation_batch_size = 240
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
decode_map = {v: k for k, v in dataset.char_dict.items()}
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate)
validation_loader = DataLoader(dataset, batch_size=validation_batch_size, sampler=valid_sampler, collate_fn=collate)
fit(model=model, epochs=10, train_data_loader=train_loader, valid_data_loader=validation_loader)


# ============================================ TESTING =================================================================
def batch_predict(model, valid_dl, up_to):
    xb, yb, lens = iter(valid_dl).next()
    model.eval()
    with torch.no_grad():
        outs = model.eager_decode(xb)
        for i in range(len(outs)):
            start = sum(lens[:i])
            end = lens[i].item()
            corr = ''.join([decode_map.get(letter.item()) for letter in yb[start:start + end]])
            predicted = ''.join([decode_map.get(letter) for letter in outs[i]])
            # ============================================ SHOW IMAGE ==================================================
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
