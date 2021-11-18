# Level3 P-stage Data Annotation


여기 밑으로 아직 수정 필요


### 💡 **Team: 컴퓨터구조**

## Project Overview

- **Predict Trash Objects**
- Input: 1024 x 1024 Image
- Output: Object annotations
    - Class(10): General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
    - Confidence score: 0 ~ 1
    - Object coordinate: (xmin, ymin, xmax, ymax)

## Archive contents

```
image-classification-level1-02/
├── dataset/
│   ├── train/ (4883 images)
│   ├── test/  (4871 images)
│   ├── train.json
│   └── test.json
├── mmdetection/
│   ├── configs/
│   ├── mmdetection library files
│   ├── train.py
│   └── inference.py
└── util/
```

- ```dataset/``` : download from [https://stages.ai/](https://stages.ai/)

## get start

### train & inference
```
cd mmdetection

python train.py
python inference.py
```

### visualize
```
cd util

jupyter notebook Visualize.ipynb

set result csv in second shell
```
it also has Visualize_val_gt -> Visualize ground truth of train data & valid

it also has Visualize_val -> Visualize result of valid

### Requirements

- Ubuntu 18.04.5
- Python 3.8.5
- pytorch 1.7.1
- torchvision 0.8.2

Install packages :  `pip install -r requirements.txt` 

#### Hardware

- CPU: 8 x Intel(R) Xeon(R) Gold 5220 CPU
- GPU: V100
- RAM: 88GB


## Contributors

| **Name** @github                                              | 
| ------------------------------------------------------------  | 
| **고재욱** [@고재욱](https://github.com/pkpete)               |
| **김성민** [@ksm0517](https://github.com/ksm0517)             |
| **박지민** [@박지민](https://github.com/ddeokbboki-good)      | 
| **박진형** [@ppjh8263](https://github.com/ppjh8263)           |
| **심세령** [@Seryoung Shim](https://github.com/seryoungshim17)| 
| **윤하정** [@Yoon Hajung](https://github.com/YHaJung)         | 

## Data Citation ```네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0```
