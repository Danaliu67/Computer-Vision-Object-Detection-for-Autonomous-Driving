import os
import tqdm
import numpy as np
import argparse
from collections import defaultdict
import pickle
from data.dataset import CAR_CLASSES, COLORS, load_json
from utils.util import *

import torch
import torchvision
from torchvision import transforms

from data.dataset import Dataset
from model.hkudetector import resnet50
from utils.loss import yololoss
from utils.LossHistory import LossHistory
import torch.optim as optim

from eval import Evaluation

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--yolo_S', default=20, type=int, help='YOLO grid num')
parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

parser.add_argument('--num_epochs', default=24, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--step_size', default=18, type=int, help='step_size')
parser.add_argument('--num_workers', default=12, type=int, help='num_workers')

parser.add_argument('--load_pretrain', default='./weights/resnet50.pth', type=str, help='pretrain weight')

parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./ass1_dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='./logs/resnet50_all', type=str, help='output directory')

parser.add_argument('--l_coord', default=5., type=float, help='hyper parameter for localization loss')
parser.add_argument('--l_noobj', default=0.5, type=float, help='hyper parameter for no object loss')

parser.add_argument('--pos_threshold', default=0.1, type=float, help='Confidence threshold for positive prediction')
parser.add_argument('--nms_threshold', default=0.5, type=float, help='Threshold for non maximum suppression')

parser.add_argument('--image_size', default=640, type=int, help='Image Size')
args = parser.parse_args()


if __name__ == '__main__':
    # ============================ step 0/5 settings  ============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    print(args)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ============================ step 1/5 数  据 ============================
    # initialize dataset
    train_dataset = Dataset(args,'trainval', transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {args.batch_size}')
    # ============================ step 2/5 模  型 ============================
    hku_mmdetector = resnet50(args=args)
    
    if args.load_pretrain is not None:
        print('Load weights {}.'.format(args.load_pretrain))
        model_dict      = hku_mmdetector.state_dict()
        # 读入预训练权重
        pretrained_dict = torch.load(args.load_pretrain, map_location = 'cpu' )
        # 筛选相同层的权值
        net_dict = hku_mmdetector.state_dict()
        for k in pretrained_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = pretrained_dict[k]
        # 载入预训练权重
        hku_mmdetector.load_state_dict(net_dict)
    hku_mmdetector = hku_mmdetector.to(device)

    # Multiple GPUs if needed
    # if torch.cuda.device_count() > 1:
    #     hku_mmdetector = torch.nn.DataParallel(hku_mmdetector)

    # =========================== step 3/5 损失函数 ===========================
    criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)

    # ============================ step 4/5 优化器 ============================
    # initialize optimizer
    optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate,weight_decay=0.0005)
    # 设置学习率下降策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # ============================ step 5/5 训  练 ============================
    Losshistory = LossHistory(output_dir)
    
    f = open(os.path.join(Losshistory.save_path, 'parameters.txt'), 'w')
    f.write(str(args))
    f.close()

    hku_mmdetector.train()

    best_val_loss = np.inf

    for epoch in range(args.num_epochs):
        hku_mmdetector.train()

        # training
        total_loss = 0.
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = hku_mmdetector(images)
            loss = criterion(pred, target)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Losshistory.append_iter_loss(loss.item())

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)

        scheduler.step() 
        Losshistory.iter_loss_plot()
        Losshistory.append_loss(total_loss/len(train_loader),total_loss/len(train_loader))

        torch.cuda.empty_cache()

    save = {'state_dict': hku_mmdetector.state_dict()}
    torch.save(save, os.path.join(Losshistory.save_path, 'hku_mmdetector_epoch'+str(args.num_epochs)+'.pth'))
