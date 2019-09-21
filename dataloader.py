from torch.utils.data import DataLoader, Dataset
import albumentations as A
from torchvision import transforms as T
from albumentations import torch as AT
import pandas as pd
import cv2
import sys
import os
import torch
from PIL import Image
import numpy as np

# for initial dataset:
# os.system('git clone https://github.com/recursionpharma/rxrx1-utils')
# sys.path.append('rxrx1-utils')
# import rxrx.io as rio

def create_transformer(transformations, images):
    target = {}
    for i, image in enumerate(images):
        target['image' + str(i)] = 'image'
    return A.Compose(transformations, p=1, additional_targets=target)(
        image=images[0],
        image1=images[1],
        image2=images[2],
        image3=images[3],
        image4=images[4],
        image5=images[5]
    )

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, mode='train', site=[1, 2], channels=[1,2,3,4,5,6], transforms=None):
        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.transforms = transforms

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    @staticmethod
    def _load_img_as_numpy(file_name):
        with Image.open(file_name) as img:
            return np.array(img).astype('float') / 255

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        site = self.site
        if not isinstance(site, int):
            site = np.random.choice(site)
        return '/'.join([self.img_dir,self.mode,experiment,'Plate{}'.format(plate),'{}_s{}_w{}.png'.format(well, site, channel)])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
#         img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        images = [self._load_img_as_numpy(img_path) for img_path in paths]

        if self.transforms:
            transform = create_transformer(self.transforms, images)
            img = torch.FloatTensor([
                transform['image'],
                transform['image1'],
                transform['image2'],
                transform['image3'],
                transform['image4'],
                transform['image5']
            ])
        else:
            img = torch.FloatTensor(images)

        if self.mode == 'train':
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

transforms = {
    'train': [
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
                               # , value=0),
            # A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
            #                     border_mode=cv2.BORDER_CONSTANT),
                                # value=0),
            A.NoOp()
        ]),
#         ZeroTopAndBottom(p=0.3),
        A.RandomSizedCrop(min_max_height=(int(512 * 0.25), int(512 * 0.75)),
                          height=256,
                          width=256, p=1.),
#         A.OneOf([
#             A.RandomBrightnessContrast(brightness_limit=0.5,
#                                        contrast_limit=0.4),
# #             IndependentRandomBrightnessContrast(brightness_limit=0.25,
# #                                                 contrast_limit=0.24),
#             A.RandomGamma(gamma_limit=(50, 150)),
#             A.NoOp()
#         ]),
#         A.OneOf([
#             FancyPCA(alpha_std=4),
#             A.HueSaturationValue(hue_shift_limit=5,
#                                  sat_shift_limit=5),
#             A.NoOp()
#         ]),
#         A.OneOf([
#             ChannelIndependentCLAHE(p=0.5),
#             A.CLAHE(),
#             A.NoOp()
#         ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ],
    'test' : [
        A.CenterCrop(height=256, width=256)
    ]
}
