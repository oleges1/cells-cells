import torch.utils.data as D
import pandas as pd
import cv2
import sys
import os
import torch

# for initial dataset:
os.system('git clone https://github.com/recursionpharma/rxrx1-utils')
sys.path.append('rxrx1-utils')
import rxrx.io as rio

# TODO: transforms

class ImagesDS(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', raw=False):
        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        # self.site = site # TODO: think what to do with this
        self.mode = mode
        self.raw = raw
        self.img_dir = img_dir
        self.len = df.shape[0]

    def load_img(self, index):
        code, experiment, plate, well = self.records[index].id_code, self.records[index].experiment, self.records[index].well, self.records[index].plate
        ims = []
        for site in [1, 2]:
            if self.raw:
                im = rio.load_site_as_rgb(
                    self.mode, experiment, plate, well, site,
                    base_path=self.img_dir
                )
                im = im.astype(np.uint8)
                # im = cv2.resize(im, self.target_shape[-1])
            else:
                save_path = os.path.join(self.img_dir, self.mode, f'{code}_s{site}.jpeg')
                im = cv2.imread(save_path)
                # im = cv2.resize(im, self.target_shape[-1])
            ims.append(torch.from_numpy(im))
        im = torch.cat(ims)
        im = im.permute(2, 1, 0)
        return im / 255. - 0.5

    def __getitem__(self, index):
        img = self.load_img(index)
        if self.mode == 'train':
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
