import pandas as pd
import os
import os.path as osp
import json
import copy
from tqdm import tqdm

add_data_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/camper')  # json 들어있는 폴더
wrong_labeled = list(pd.read_csv('/opt/ml/code/utils/etc/revised_wrong0~500.csv')['id'])

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('revised_train_original_wrongcutpolygon')), 'r') as f:
    anno = json.load(f)

anno = anno['images']

anno_temp = copy.deepcopy(anno)

for img_name, img_info in tqdm(anno.items()) :
    # 라벨링 없는 경우/잘못 라벨링된 경우 삭제
    if img_info['words'] == {} or img_name in wrong_labeled:
        del(anno_temp[img_name])
        print(f'{img_name} deleted')
        continue
        
anno = {'images': anno_temp}

with open(osp.join(add_data_dir,'ufo', 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)