from os.path import join
import os
import yaml
from utils import *
from dataloader import *
from models import *


config = DotDict(yaml.load(join(name, 'config.yaml')))

def train(config, num_classes=1108):
    model = model_whale(num_classes=num_classes, inchannels=6, model_name=config.train.model_name).cuda()
    i = 0
    iter_smooth = 50
    iter_valid = 200
    iter_save = 200
    epoch = 0
    if config.train.freeze:
        model.freeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr,  betas=(0.9, 0.99), weight_decay=0.0002)

    resultDir = join(name, '{}_{}'.format(model_name, fold_index))
    checkPoint = join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)

    train_dataset = ImagesDS(config.train.csv_file, config.train.img_dir)
    # val_dataset = ImagesDS(config.train.val_csv_file, config.train.val_img_dir)
    dataloader_train = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=config.train.batch_size, num_workers=config.train.num_workers)
    # dataloader_val = DataLoader(val_dataset, shuffle=True, drop_last=False, batch_size=config.train.batch_size, num_workers=config.train.num_workers)
