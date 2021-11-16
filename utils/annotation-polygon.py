import os.path as osp
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)
        
add_data_dir = os.environ.get('SM_CHANNEL_TRAIN', '../input/data/dataset_revised')

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('annotation')), 'r') as f:
# 여기 부분도 가지고 계신 파일명에 맞추어서 수정해주시길 요청드립니다.
    anno = json.load(f)

anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for img_name, img_info in tqdm(anno.items()) :
    # 라벨링 없는 경우 삭제
    if img_info['words'] == {}:
        del(anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items() :
        # polygon 점 4개인 경우 pass
        if len(img_info['words'][obj]['points']) == 4 :
            count_normal += 1
            continue
        # polygon 점 4개 이하인 경우 delete
        elif len(img_info['words'][obj]['points']) < 4 :
            del(anno_temp[img_name]['words'][obj])
        else :
            # 폴리곤을 사각단위로 잘라서 각 부분을 word의 영역으로 사용
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            point_len = len(img_info['words'][obj]['points'])
            for index in range(len(img_info['words'][obj]['points'])//2-1):
                over_poly_region = []
                over_poly_region.append(img_info['words'][obj]['points'][index])
                over_poly_region.append(img_info['words'][obj]['points'][index+1])
                over_poly_region.append(img_info['words'][obj]['points'][point_len-2-index])
                over_poly_region.append(img_info['words'][obj]['points'][point_len-1-index])
                over_polygon_temp['points'] = over_poly_region
                anno_temp[img_name]['words'][obj+f'{index+911}'] = copy.deepcopy(over_polygon_temp)
            del anno_temp[img_name]['words'][obj]
        if anno_temp[img_name]['words'] == {} :
            del(anno_temp[img_name])
            count_none_anno += 1
            continue
        count += 1
            
print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')
print(count_none_anno)

# np_point = np.array(img_info['words'][obj]['points'])
# starting_point = np_point[0]

anno = {'images': anno_temp}

ufo_dir = osp.join(add_data_dir, 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)