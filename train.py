from os.path import join
import time
import os
import yaml
import torch
from utils import *
from dataloader import *
from models import *
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from radam import RAdam
from lookahead import Lookahead

def valid_eval(config, model, dataLoader_valid):
    with torch.no_grad():
        if config.train.enable_eval_on_val:
            model.eval()
            model.mode = 'valid'
        top1_batch = 0.
        map5_batch = 0.
        loss = 0.
        for i, valid_data in enumerate(dataLoader_valid):
            images, labels = valid_data
            images = images.cuda()
            labels = labels.cuda().long()
            global_feat, local_feat, results = data_parallel(model, images)
            model.getLoss(global_feat, local_feat, results, labels, config, verbose=(i % config.loss.verbose_interval == 0))
            loss += model.loss
            results = torch.sigmoid(results)
            top1_batch += accuracy(results, labels, topk=[1])[0]
            map5_batch += mapk(labels, results, k=5)
        return loss / len(dataLoader_valid), top1_batch / len(dataLoader_valid), map5_batch / len(dataLoader_valid)

def train(config, num_classes=1108):
    model = model_whale(num_classes=num_classes, inchannels=6, model_name=config.train.model_name, pretrained=config.train.pretrained).cuda()
    if config.train.freeze:
        model.freeze()

    base_opt = RAdam(model.parameters(), lr=config.train.lr)
    optimizer = Lookahead(base_opt)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr,  betas=(0.9, 0.99), weight_decay=0.0002)

    resultDir = config.train.result_dir
    checkPoint = join(resultDir, 'checkpoint')
#     if not config.train.in_colab:
#         os.makedirs(checkPoint, exist_ok=True)
    train_dataset = CustomDataset(config.train.csv_file, config.train.img_dir, transforms=transforms['train'])
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.train.validation_split * dataset_size))
    if config.train.shuffle_dataset:
        np.random.seed(config.train.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                               sampler=train_sampler, num_workers=config.train.num_workers)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                                    sampler=valid_sampler, num_workers=config.train.num_workers)

    train_loss = 0.

    # load from cpk:
    if config.train.load_cpk:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (config.train.start_epoch)),skip=[])
        cpk = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (config.train.start_epoch)))
        optimizer.load_state_dict(cpk['optimizer'])
        adjust_learning_rate(optimizer, config.train.lr)
        start_epoch = cpk['epoch']
    else:
        start_epoch = 0

    top1_batch, map5_batch = 0, 0

    for epoch in range(start_epoch + 1, config.train.epochs):
        print('Starting:', epoch, 'Iterations:', len(train_loader))
        for i, data in enumerate(train_loader):
            model.train()
            model.mode = 'train'
            images, labels = data
            images = images.cuda()
            labels = labels.cuda().long()
            global_feat, local_feat, results = data_parallel(model, images)
            model.getLoss(global_feat, local_feat, results, labels, config, verbose=(i % config.loss.verbose_interval == 0))
            batch_loss = model.loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            results = torch.sigmoid(results)
            train_loss += batch_loss.data.cpu().numpy()
            top1_batch += accuracy(results, labels, topk=[1])[0]
            map5_batch += mapk(labels, results, k=5)

            if i % config.train.verbose_interval == 0:
                print('epoch: %03d, iter: %05d, train_loss: %f, top1_batch: %f, map5_batch: %f' % (epoch, i, float(train_loss / config.train.verbose_interval), float(top1_batch / config.train.verbose_interval), float(map5_batch / config.train.verbose_interval)))
                
#                 print(f'epoch: {epoch}, iter: {i}, train_loss: {float(train_loss / config.train.verbose_interval)}, top1_batch: {float(top1_batch / config.train.verbose_interval)}, map5_batch: {float(map5_batch / config.train.verbose_interval)}')
                train_loss, top1_batch, map5_batch = 0, 0, 0

                valid_loss, top1_valid, map5_valid = valid_eval(config, model, validation_loader)
                print('epoch: %03d, iter: %05d, valid_loss: %f, valid_top1_batch: %f, valid_map5_batch: %f' % (epoch, i, valid_loss, top1_valid, map5_valid))
#                 print(f'epoch: {epoch}, iter: {i}, valid_loss: {valid_loss}, top1_batch: {top1_valid}, map5_batch: {map5_valid}')



        if epoch % config.train.save_period == 0:
            os.system("touch " + resultDir + "/checkpoint/%08d_model.pth" % (epoch))
            os.system("touch " + resultDir + "/checkpoint/%08d_optimizer.pth" % (epoch))
            time.sleep(1)
            torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (epoch))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, resultDir + '/checkpoint/%08d_optimizer.pth' % (epoch))

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = DotDict(yaml.load(f))
    train(config)
