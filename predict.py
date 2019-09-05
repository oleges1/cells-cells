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
import numpy as np
import pandas as pd
from tqdm import tqdm

def predict_model(config, num_classes=1108):
    test_dataset = ImagesDS(config.test.csv_file, config.test.img_dir, mode='test')
    dataloader_test = DataLoader(test_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers)
    model = model_whale(num_classes=num_classes, inchannels=12, model_name=config.train.model_name).cuda()
    model.load_pretrain(os.path.join(config.test.checkpoints_path, '%08d_model.pth' % (config.test.epoch)), skip=[])
    result = {}
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            images = images.cuda()
            _, _, outs = data_parallel(model, images)
            outs = torch.sigmoid(outs)
            for name, out in zip(names, outs):
                result[name] = out.cpu().numpy()
    test_csv = pd.read_csv(config.test.csv_file)
    test_csv['result'] = test_csv['id_code'].map(result)
    return np.vstack(test_csv['result'].values)

def select_plate_group(idx, pp_mult, all_test_exp, test_csv, exp_to_group, plate_groups):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
           np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult

def leak_postprocess(config, predicts):
    train_csv = pd.read_csv(config.train.csv_file)
    test_csv = pd.read_csv(config.test.csv_file)

    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum()

    all_test_exp = test_csv.experiment.unique()
    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds = predicts[test_csv.experiment == all_test_exp[idx]]
        pp_mult = np.zeros((len(preds),1108))
        pp_mult[range(len(preds)),preds.argmax(axis=-1)] = 1

        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                   np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
    exp_to_group = group_plate_probs.argmax(1)

    result = np.zeros((len(test_csv), ))
    for idx in range(len(all_test_exp)):
        #print('Experiment', idx)
        indices = (test_csv.experiment == all_test_exp[idx])
        pp_mult = predicts[test_csv.experiment == all_test_exp[idx]]
        preds = select_plate_group(idx, pp_mult, all_test_exp, test_csv, exp_to_group, plate_groups)
        result[indices] = preds.argmax(1)
    return result


def save_csv(config, predicts):
    submission = pd.read_csv(config.test.csv_file)
    submission['sirna'] = predicts.astype(int)
    submission.to_csv(config.test.save_path, index=False, columns=['id_code', 'sirna'])

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = DotDict(yaml.load(f))
    predicts = predict_model(config)
    predicts = leak_postprocess(config, predicts)
    save_csv(config, predicts)
