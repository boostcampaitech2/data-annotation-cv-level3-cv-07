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

from shapely import affinity
from shapely.geometry import Polygon


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']

# determinate TTA here!!
# 여기서 TTA를 무엇을 할지 정해서 넣어줍니다.
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[1, 2]),
    ]
)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    # parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL')) 
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean')) 
    
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'adamW_models'))# trained_models
    
    # parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions')) 
    parser.add_argument('--output_dir', default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean')) 
    
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

# 앙상블은 위한 코드 -> 이 부분 조금 더 고민을 해볼 필요성이 있어보입니다.
'''
Need_to_discuss!
기존 nms는 score을 기준으로 남겨주었는데 이번꺼는 기준이 크기가 큰 것이기 때문에 장단점이 있습니다.
장 - 글자 전체를 가져갈 확률이 높음, 상대적으로 글자가 있으면 예측해줄 확률이 높아짐
단 - 크기가 크면 단어가 아닌 영역을 가져갈 확률이 높아짐, 글자를 가져갔어도 중간에 공백인 부분을 가져갈 확률이 높아짐
이 모델 한정 심지어 음수를 예측해주기도 합니다. 따라서 단점이 생각보다 크게 다가올 수 있는 영역입니다.
'''
def ensemble(bboxes, iou_threshold) :
    # 살릴 박스를 정하는 부분
    keep = np.ones(len(bboxes))
    # 모든 박스를 돌면서
    for idx, bbox in enumerate(bboxes) :
        # 자기 뒤로 모든 박스와 비교를 함
        # if keep[idx] != 0 :
        for idx_compare, compare in enumerate(bboxes[idx+1:]) :
            # iou를 구하고
            polygon1 = Polygon(bbox)
            polygon2 = Polygon(compare)
            intersect = polygon1.intersection(polygon2).area
            union = polygon1.union(polygon2).area
            iou = intersect / union
            # iou threshold를 넘기면 1개는 지워줌(idx큰거 우선)
            if iou > iou_threshold :
                if polygon1.area > polygon2.area :
                    keep[idx+idx_compare+1] = 0
                else :
                    keep[idx] = 0
        # else :
        #     continue
    # keep에 따라서 박스들을 지워나감
    bboxes = bboxes[np.where(keep==1)]
    return bboxes

# 정확 - 이미지의 가로 중간축을 기준으로 bbox를 회전 시키는 함수
def flip_horizontal_bbox(bbox, image_size) :
    image_w = image_size[0]/2
    bbox_temp = np.zeros_like(bbox)
    for i, point in enumerate(bbox) :
        bbox_temp[i][0] = 2*image_w - point[0]
        bbox_temp[i][1] = point[1]
    return bbox_temp

# 정확 - 이미지의 중심을 기준으로 bbox를 회전 시키는 함수
def rotate_bbox(bbox, theta, image_size):
    k = theta / 90
    # 회전의 횟수만큼 회전행렬 변환을 통해 만들어진 식으로 회전)
    bbox_temp = np.zeros_like(bbox)
    if k==1 :
        for i, point in enumerate(bbox) :
            bbox_temp[i][0] = round(point[1] -image_size[1]/2 + image_size[0]/2, 4)
            bbox_temp[i][1] = round(image_size[0]/2 - point[0] + image_size[1]/2, 4)
    elif k==2 :
        for i, point in enumerate(bbox) :
            bbox_temp[i][0] = round(image_size[0]/2 - point[0] + image_size[0]/2, 4)
            bbox_temp[i][1] = round(image_size[1]/2 - point[1] + image_size[1]/2, 4)
    else :
        for i, point in enumerate(bbox) :
            bbox_temp[i][0] = round(image_size[1]/2 -point[1] + image_size[0]/2, 4)
            bbox_temp[i][1] = round(point[0] - image_size[0]/2 + image_size[1]/2, 4)
    return bbox_temp

# 정확 - scale을 2배 줄여서 기존의 2배한 aug를 원상 복귀
def scale_back_bbox(bbox):
    return bbox/2

# 작업하다 tta는  불필요하다고 느껴서 작업 멈춤
# def delete_over_size(detected_bbox, image_size) :
#     box_list_temp = np.zeros_like
#     for i, detected_box in enumerate(detected_bbox) :
        

# 아직 출력시 405.89563 이런식이여야 하는 데 4.058956345e+02같은 문제가 잔존 제출시 이 부분이 점수 감점일 듯
def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='public'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    
    # 지수표기를 막아줍니다.
    np.set_printoptions(precision=6, suppress=True)
    
    for image_fpath in tqdm(glob(osp.join(data_dir, '{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))
        
        # augmentation적용을 위해 형변환
        image = Image.open(image_fpath)
        image_size = image.size
        image = np.array(image) / 255  # 이미지를 읽고 min max scaling
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
                # for문 2개를돌면서 박스를 다시 deaug해줍니다.
                for idx, detected_box_img in enumerate(detected_box) :
                    for i, one_box in enumerate(detected_box_img) :
                        if (idx+1) % 2 == 0 :
                            detected_box_img[i] = scale_back_bbox(one_box)
                            if (idx+1) % 8 == 2 or (idx+1) % 8 == 3 :
                                detected_box_img[i] = rotate_bbox(one_box, 90, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                            if (idx+1) % 8 == 3 or (idx+1) % 8 == 4 :
                                detected_box_img[i] = rotate_bbox(one_box, 180, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                            if (idx+1) % 8 == 5 or (idx+1) % 8 == 6 :
                                detected_box_img[i] = rotate_bbox(one_box, 270, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                        else :
                            if (idx+1) % 8 == 2 or (idx+1) % 8 == 3 :
                                detected_box_img[i] = rotate_bbox(one_box, 90, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                            if (idx+1) % 8 == 3 or (idx+1) % 8 == 4 :
                                detected_box_img[i] = rotate_bbox(one_box, 180, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                            if (idx+1) % 8 == 5 or (idx+1) % 8 == 6 :
                                detected_box_img[i] = rotate_bbox(one_box, 270, image_size)
                                if (idx) >= 8 : detected_box_img[i] = flip_horizontal_bbox(one_box, image_size)
                    detected_box[idx] = detected_box_img
                    
                # list로 되어 있는 부분을 np.array로 변경시켜줍니다.
                detected_box = np.array(detected_box, dtype='object')
                # 한 이미지에서 나온 것이므로 합쳐서 보내줍니다.
                detected_box = np.concatenate(detected_box, axis=0)
                # 
                # 여기서 나온 box결과들을 앙상블 하고
                detected_box = ensemble(detected_box, iou_threshold=0.1)                
                print(detected_box)
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
