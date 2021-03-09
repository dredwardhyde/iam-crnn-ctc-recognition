import os
import sys

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from colorama import Fore
from skimage import io, transform, color
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from tqdm import tqdm


def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return img


class IAMData(Dataset):

    def __init__(self, txt_file, root_dir, output_size, border_pad=(0, 0)):
        gt = []
        print("Preparing dataset...")
        # Open raw lines.txt
        for line in open(txt_file):
            # Ignore comments
            if not line.startswith("#"):
                # Split each string by whitespaces
                info = line.strip().split()
                # If string was recognized correctly
                if info[1] == 'ok':
                    # First column is filename, second column is target sentence
                    gt.append((info[0] + '.png', ' '.join(info[8:]).replace('|', ' ').lower()))

        # Convert target array to Dataframe
        df = pd.DataFrame(gt, columns=['file', 'word'])
        self.line_df = df
        # Compute char set
        chars = []
        # Take all values from the last column, convert to chars and add them to array
        self.line_df.iloc[:, -1].apply(lambda x: chars.extend(list(x)))
        self.all_lines = ' '
        # Remove duplicates
        chars = sorted(list(set(chars)))
        # Convert to dictionary like {1:'c'}
        self.char_dict = {c: i for i, c in enumerate(chars)}
        # Convert raw data to dataset
        self.samples = {}
        progress_bar = tqdm(total=len(self.line_df),
                            position=0, leave=True,
                            file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        for idx in range(0, len(self.line_df)):
            # Take image filename from the first column
            img_name = self.line_df.iloc[idx, 0]
            # Split filename by dash to get the full path
            im_nm_split = img_name.split('-')
            # Get the first level folder
            start_folder = im_nm_split[0]
            # Get the second level folder
            src_folder = '-'.join(im_nm_split[:2])
            # Target folder is ready
            folder_name = os.path.join(start_folder, src_folder)
            # Calculate the full path to the image
            img_filepath = os.path.join(root_dir,
                                        folder_name,
                                        img_name)
            # Read target image
            image = io.imread(img_filepath)
            # Read target sentence
            word = self.line_df.iloc[idx, -1]
            self.all_lines = self.all_lines + ' ' + word
            # Calculate resulting size like target_size - borders
            resize = (output_size[0] - border_pad[0], output_size[1] - border_pad[1])
            # Get height and width of the image
            h, w = image.shape[:2]
            # Calculate width and height scaling to the target size
            fx = w / resize[1]
            fy = h / resize[0]
            # Get the maximum scale
            f = max(fx, fy)
            # Calculate new size like either maximum target dimension or scaled one min(resize[0], int(h / f)
            # max(min(resize[0], int(h / f)), 1) etc. sets the lower bound
            new_size = (max(min(resize[0], int(h / f)), 1), max(min(resize[1], int(w / f * np.random.uniform(1, 3))), 1))
            # Resize image and fill all empty area with white color
            image = transform.resize(image, new_size, preserve_range=True, mode='constant', cval=255)
            image = deskew(image)
            # Prepare canvas for the resized image
            canvas = np.ones(output_size, dtype=np.uint8) * 255
            # Calculate maximum actual padding
            v_pad_max = output_size[0] - new_size[0]
            h_pad_max = output_size[1] - new_size[1]
            # Generate new padding to place image somewhere between 0 and pad_max
            v_pad = int(np.random.choice(np.arange(0, v_pad_max + 1), 1))
            h_pad = int(np.random.choice(np.arange(0, h_pad_max + 1), 1))
            # Place image
            canvas[v_pad:v_pad + new_size[0], h_pad:h_pad + new_size[1]] = image
            # Rotate image 90 degrees counter-clockwise
            canvas = transform.rotate(canvas, -90, resize=True)[:, :-1]
            # Convert to RGB from greyscale
            canvas = color.gray2rgb(canvas)
            # Transpose tensor
            canvas = torch.from_numpy(canvas.transpose((2, 0, 1))).float()
            # Normalize image
            norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            sample = {'image': norm(canvas), 'word': word}
            progress_bar.update(1)
            self.samples[idx] = sample
        progress_bar.close()

    def __len__(self):
        return len(self.line_df)

    def __getitem__(self, idx):
        return self.samples[idx]
