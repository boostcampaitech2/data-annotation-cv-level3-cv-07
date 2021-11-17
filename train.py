import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--add_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/camper'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'adamW_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=131)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args
    
def do_training(data_dir, add_data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    data_dir_list = [data_dir, add_data_dir] # data를 리스트로 받아옴
    dataset = SceneTextDataset(data_dir_list, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # AdamW고정 필요
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1) # cos annealing 필요 (restart?)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)

    wandb.watch(model) 
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                wandb.log({
                    "train/loss" : loss_val,
                    "train/Cls_loss": extra_info['cls_loss'],
                    "train/Angle_loss" : extra_info['angle_loss'],
                    "train/IoU_loss" : extra_info['iou_loss'],
                    })
                pbar.set_postfix(val_dict)
            wandb.log({
                "epoch/loss" : epoch_loss / num_batches,
                "epoch" : epoch + 1,
                'learning_rate': scheduler.optimizer.param_groups[0]['lr']
            })
        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
    wandb.finish()


def main(args):
    wandb.init(
        project='Data_anno_ocr',
        #   name=args.model_name
    )
    wandb.config.update(args)

    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
