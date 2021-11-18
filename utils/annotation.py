import os.path as osp
import json
import os
from tqdm import tqdm
import numpy as np
import copy

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)
        
add_data_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/camper')
# 여기 부분 가지고 계신 폴더구성에 맞추어 수정해주시구요

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('annotation')), 'r') as f:
# 여기 부분도 가지고 계신 파일명에 맞추어서 수정해주시길 요청드립니다.
    anno = json.load(f)

anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for img_name, img_info in tqdm(anno.items()) :
    if img_info['words'] == {} :
        del(anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items() :
        # illegibility는 전부 단어 이므로 false
        anno_temp[img_name]['words'][obj]['illegibility'] = False
        if len(img_info['words'][obj]['points']) == 4 :
            count_normal += 1
            continue
            
        elif len(img_info['words'][obj]['points']) < 4 :
            del(anno_temp[img_name]['words'][obj])
        # 폴리곤 수정시에는 여기 부분을 수정해주시면 됩니다!!
        # 다음 예제는 polygon이 넘칠 경우 해당 폴리곤을 illegibility를 삭제처리
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
        else :
            # 현동님의 기여로 만들어진 부분
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            over_poly_region = copy.deepcopy(over_polygon_temp)
            over_poly_region['points'] = []
            for index in range(len(img_info['words'][obj]['points'])//2 -1):
                over_poly_region['points'].append(over_polygon_temp['points'][index])
                over_poly_region['points'].append(over_polygon_temp['points'][index+1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index-1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index])
                anno_temp[img_name]['words'][obj+f'{index+911}'] = copy.deepcopy(over_poly_region) #911 현동님 생일 >_<
                over_poly_region['points'] = []
            del anno_temp[img_name]['words'][obj]
            # 폴리곤을 사각단위로 잘라서 각 부분을 word의 영역으로 사용하는 코드입니다.
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

ufo_dir = osp.join('/opt/ml/input/data/camper', 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train_new.json'), 'w') as f:
    json.dump(anno, f, indent=4)