import pandas as pd
import numpy as np
import torch
from skimage import io, transform, color
import os
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class IAMData(Dataset):

    def __init__(self, txt_file, root_dir, output_size, border_pad=(0, 0), random_rotation=0, random_stretch=1):
        gt = []
        for line in open(txt_file):
            if not line.startswith("#"):
                info = line.strip().split()
                if info[1] == 'ok':
                    gt.append((info[0] + '.png', info[1], ' '.join(info[8:]).replace('|', ' ')))

        df = pd.DataFrame(gt, columns=['file', 'ok', 'word'])
        self.line_df = df
        chars = []
        self.line_df.iloc[:, -1].apply(lambda x: chars.extend(list(x)))
        chars = sorted(list(set(chars)))
        self.char_dict = {c: i for i, c in enumerate(chars, 1)}
        self.max_len = self.line_df.iloc[:, -1].apply(lambda x: len(x)).max()
        self.samples = {}
        for idx in range(0, len(self.line_df)):
            img_name = self.line_df.iloc[idx, 0]
            im_nm_split = img_name.split('-')
            start_folder = im_nm_split[0]
            src_folder = '-'.join(im_nm_split[:2])
            folder_name = os.path.join(start_folder, src_folder)
            img_filepath = os.path.join(root_dir,
                                        folder_name,
                                        img_name)
            image = io.imread(img_filepath)
            word = self.line_df.iloc[idx, -1]
            resize = (output_size[0] - border_pad[0], output_size[1] - border_pad[1])
            h, w = image.shape[:2]
            fx = w / resize[1]
            fy = h / resize[0]
            f = max(fx, fy)
            new_size = (max(min(resize[0], int(h / f)), 1), max(min(resize[1], int(w / f * random_stretch)), 1))
            image = transform.resize(image, new_size, preserve_range=True, mode='constant', cval=255)
            rot = np.random.choice(np.arange(-random_rotation, random_rotation), 1)
            image = transform.rotate(image, rot, mode='constant', cval=255, preserve_range=True)

            canvas = np.ones(output_size, dtype=np.uint8) * 255

            v_pad_max = output_size[0] - new_size[0]
            h_pad_max = output_size[1] - new_size[1]
            v_pad = int(np.random.choice(np.arange(0, v_pad_max + 1), 1))
            h_pad = int(np.random.choice(np.arange(0, h_pad_max + 1), 1))

            canvas[v_pad:v_pad + new_size[0], h_pad:h_pad + new_size[1]] = image
            canvas = transform.rotate(canvas, -90, resize=True)[:, :-1]
            canvas = color.grey2rgb(canvas)
            canvas = torch.from_numpy(canvas.transpose((2, 0, 1))).float()

            norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            sample = {'image': norm(canvas), 'word': word}
            self.samples[idx] = sample

    def __len__(self):
        return len(self.line_df)

    def __getitem__(self, idx):
        return self.samples[idx]