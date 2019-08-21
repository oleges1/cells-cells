# TODO: predict here using leak
# TODO: test-time augmentations

from os.path import join
import os
import yaml
import torch
from utils import *
from dataloader import *
from models import *
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.dataloader import DataLoader

def predict_model(config):
    test_dataset = ImagesDS(config.test.csv_file, config.test.img_dir, mode='test')
    dataloader_test = DataLoader(test_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers)
    model = model_whale(num_classes=num_classes, inchannels=6, model_name=config.train.model_name).cuda()
    model.load_pretrain(os.path.join(config.test.checkpoints_path, '%08d_model.pth' % (config.test.epoch)), skip=[])
    result = defaultdict(int)
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            images = images.cuda()
            _, _, outs = model(images)
            outs = torch.sigmoid(outs)
            for name, out in zip(names, outs):
                result[name] = int(out.argmax())
    return result

def leak_postprocess(config, predicts):
    pass

def save_csv(config, predicts):
    pass

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = DotDict(yaml.load(f))
    predicts = predict_model(config)
    predicts = leak_postprocess(config, predicts)
    save_csv(config, predicts)
