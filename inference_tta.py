import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect

import ttach as tta
import numpy as np
from PIL import Image
import torch
from shapely.geometry import Polygon
import lanms


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']

# determinate TTA here!!
# 여기서 TTA를 무엇을 할지 정해서 넣어줍니다.
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[1, 2]),
        # tta.FiveCrops(384, 384),
        # tta.Multiply(factors=[0.7, 1, 1.3]),
    ]
)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL')) 
    # default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean') # default=os.environ.get('SM_CHANNEL_EVAL')
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'revised_models'))# trained_models
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions')) 
    # default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean') # default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions')
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

# 앙상블은 위한 코드
def ensemble(bboxes, iou_threshold):
    # 살릴 박스를 정하는 부분
    keep = np.ones(len(bboxes))
    # 모든 박스를 돌면서
    for idx, bbox in enumerate(bboxes) :
        # 자기 뒤로 모든 박스와 비교를 함
        for compare in bboxes[idx+1:] :
            # iou를 구하고
            polygon1 = Polygon(bbox)
            polygon2 = Polygon(compare)
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersect / union
            # iou threshold를 넘기면 1개는 지워줌(idx큰거 우선)
            if iou > iou_threshold :
                keep[idx+1] = 0
    # keep에 따라서 박스들을 지워나감
    bboxes = bboxes[np.where(keep==1)]
    return bboxes 

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='public'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    for image_fpath in tqdm(glob(osp.join(data_dir, '{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))
        
        # augmentation적용을 위해 형변환
        image = np.array(Image.open(image_fpath)) / 255  # 이미지를 읽고 min max scaling
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        
        count = 0
        detected_box = []
        for transformer in transforms:  # custom transforms (transform을 한개씩 불러와줍니다.)
            auged_image = transformer.augment_image(image) # make image to augmented (입혀줍니다.) # cv2.imread(image_fpath)[:, :, ::-1]
            auged_image = np.array((auged_image*255).squeeze()).transpose(1, 2, 0).astype(np.uint8) # 다시 형변환을 풀어서
            images.append(auged_image) # append augmented image (넣어줍니다.)
            # 총 트랜스폼의 수와 같다면
            if len(images) == len(transforms) :
                detected_box = detect(model, images, input_size) # 하나의 배치처럼 전부 이미지를 infe하고
                detected_box = np.concatenate(detected_box, axis=0)
                print(len(detected_box))
                # 여기서 나온 box결과들을 앙상블 하고
                detected_box = ensemble(detected_box, iou_threshold=0.5)
                print(len(detected_box))
                # 앙상블된 결과를 추가해준다.
                by_sample_bboxes.append(detected_box)
                images = []
        
        # if len(images):
        #     by_sample_bboxes.extend(detect(model, images, input_size))
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'epoch_130.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    for split in ['public', 'private']:
        print('Split: {}'.format(split))
        split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                    args.batch_size, split=split)
        ufo_result['images'].update(split_result['images'])

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
