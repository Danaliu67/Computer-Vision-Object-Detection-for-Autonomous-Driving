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
parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

parser.add_argument('--num_epochs', default=24, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--step_size', default=18, type=int, help='step_size')
parser.add_argument('--num_workers', default=8, type=int, help='num_workers')

parser.add_argument('--load_pretrain', default='./weights/resnet50.pth', type=str, help='pretrain weight')

parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./ass1_dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='./logs/resnet50', type=str, help='output directory')

parser.add_argument('--l_coord', default=5., type=float, help='hyper parameter for localization loss')
parser.add_argument('--l_noobj', default=0.5, type=float, help='hyper parameter for no object loss')

parser.add_argument('--pos_threshold', default=0.1, type=float, help='Confidence threshold for positive prediction')
parser.add_argument('--nms_threshold', default=0.5, type=float, help='Threshold for non maximum suppression')

parser.add_argument('--image_size', default=448, type=int, help='Image Size')
args = parser.parse_args()

if __name__ == '__main__':
    # ============================ step 0/5 settings  ============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    print(args)

    # output_dir = args.output_dir
    output_dir = os.path.join(args.output_dir, 'Size'+str(args.image_size),
                              'Batch'+str(args.batch_size),'epochs'+str(args.num_epochs))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ============================ step 1/5 数  据 ============================
    # initialize dataset
    train_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = Dataset(args, split='val', transform=[transforms.ToTensor()])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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

    # =========================== step 3/5 损失函数 ===========================
    criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)

    # ============================ step 4/5 优化器 ============================
    # initialize optimizer
    optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate,weight_decay=0.0005)
    # 设置学习率下降策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # ============================ step 5/5 训  练 ============================
    Losshistory = LossHistory(output_dir)
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
        # validation
        validation_loss = 0.0
        hku_mmdetector.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = hku_mmdetector(images)
                loss = criterion(prediction, target)
                validation_loss += loss.data
            validation_loss /= len(val_loader)
            print("validation loss:", validation_loss.item())

        Losshistory.append_loss(total_loss/len(train_loader),validation_loss.item())
        Losshistory.iter_loss_plot()

        if best_val_loss > validation_loss:
            best_val_loss = validation_loss

            save = {'state_dict': hku_mmdetector.state_dict()}
            torch.save(save, os.path.join(Losshistory.save_path, 'hku_mmdetector_best.pth'))

        # save = {'state_dict': hku_mmdetector.state_dict()}
        # torch.save(save, os.path.join(Losshistory.save_path, 'hku_mmdetector_epoch_'+str(epoch+1)+'.pth'))

        torch.cuda.empty_cache()

    # ============================ step 6/5 验  证 ============================

    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('DATA PREPARING...')
    annotation_path = os.path.join(args.dataset_root, 'annotations', 'instance_val.json')
    annotations = load_json(annotation_path)

    for annotation in annotations['annotations']:
        image_name = annotation['image_name']
        if image_name not in image_list:
            image_list.append(image_name)
        bbox = annotation['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        c = int(annotation['category_id'])
        class_name = CAR_CLASSES[c-1]
        targets[(image_name, class_name)].append([x1, y1, x2, y2])
    print('DONE.')
    print('START EVALUATION...')

    model = hku_mmdetector


    model.load_state_dict(torch.load(os.path.join(Losshistory.save_path, 'hku_mmdetector_best.pth'))['state_dict'])
    model.eval()
    with torch.no_grad():
        for image_name in  tqdm.tqdm(image_list):
            image_path = os.path.join(args.dataset_root, 'val', 'image', image_name)
            result = inference(args, model, image_path)

            for (x1, y1), (x2, y2), class_name, image_name, conf in result:

                predictions[class_name].append([image_name, conf, x1, y1, x2, y2])

    # write the prediction result
    f = open(os.path.join(Losshistory.save_path, 'result.pkl'), 'wb')
    pickle.dump(args, f)
    pickle.dump(predictions, f)
    f.close()

    print('BEGIN CALCULATE MAP...')
    aps = Evaluation(predictions, targets, threshold=args.pos_threshold).evaluate()
    print(f'mAP: {np.mean(aps):.2f}')
    f = open(os.path.join(Losshistory.save_path, 'parameters.txt'), 'w')
    f.write(str(args))
    f.write(str(aps))
    f.write(f'mAP: {np.mean(aps):.2f}')
    f.close()
    print('DONE.')