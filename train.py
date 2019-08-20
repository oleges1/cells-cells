from os.path import join
import os
import yaml
import torch
from utils import *
from dataloader import *
from models import *
from torch.nn.parallel.data_parallel import data_parallel

with open('config.yaml', 'r') as f:
    config = DotDict(yaml.load(f))

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

    resultDir = config.train.result_dir
    checkPoint = join(resultDir, 'checkpoint')
    if not config.train.in_colab:
        os.makedirs(checkPoint, exist_ok=True)

    train_dataset = ImagesDS(config.train.csv_file, config.train.img_dir)
    # val_dataset = ImagesDS(config.train.val_csv_file, config.train.val_img_dir)
    dataloader_train = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=config.train.batch_size, num_workers=config.train.num_workers)
    # dataloader_val = DataLoader(val_dataset, shuffle=True, drop_last=False, batch_size=config.train.batch_size, num_workers=config.train.num_workers)

    train_loss = 0.

    # load from cpk:
    if config.train.load_cpk:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (config.train.start_epoch)),skip=skips)
        cpk = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (config.train.start_epoch)))
        optimizer.load_state_dict(cpk['optimizer'])
        adjust_learning_rate(optimizer, lr)
        start_epoch = cpk['epoch']
    else:
        start_epoch = 0

    top1_batch, map5_batch = 0, 0

    for epoch in range(start_epoch, config.train.epochs):
        for i, data in enumerate(dataloader_train):
            model.train()
            images, labels = data
            images = images.cuda()
            labels = labels.cuda().long()
            global_feat, local_feat, results = data_parallel(model, images)
            model.getLoss(global_feat, local_feat, results, labels)
            batch_loss = model.loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
            train_loss += batch_loss.data.cpu().numpy()
            top1_batch += accuracy(results, labels, topk=(1,))[0]
            map5_batch += mapk(labels, results, k=5)

            if i % config.train.verbose_interval == 0 and not epoch == start_epoch:
                print(f'epoch: {epoch}, iter: {i}, top1_batch: {top1_batch / config.train.verbose_interval}, map5_batch: {map5_batch / config.train.verbose_interval}')
                train_loss, top1_batch, map5_batch = 0, 0, 0


        if epoch % config.train.save_period == 0 and not epoch == start_epoch:
            torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (epoch))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, resultDir + '/checkpoint/%08d_optimizer.pth' % (epoch))

if __name__ == "__main__":
    train(config)
