import torch
from torch.utils.data import DataLoader, Dataset
import os.path as osp
import pandas as pd
from PIL import Image
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir):
        super().__init__()
        self.data_dir = data_dir # data 경로 폴더
        self.coco = COCO(annotation) # coco annotation 불러오기 (coco API)

    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image = osp.join(self.data_dir, image_info['file_name'])

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
    
# 이미지의 예측된 box 얻어서 submission file 내용 만들기
def get_bboxes(annotation, data_loader, model):

    
    # submission 파일에 저장될 내용
    prediction_strings = []
    file_names = [] 
    
    coco = COCO(annotation)
    for batch_idx, image_batch in enumerate(tqdm(data_loader)):
        # 이미지에서 예측한 모든 박스들
        all_pred_boxes = []
        
        #이미지 정보
        
        with torch.no_grad():
            predictions = model(image_batch)
        batch_size = len(image_batch)
        for idx in range(batch_size):
            file_name = osp.join('test', osp.basename(image_batch[idx]))
            pred_info = predictions[idx].boxes
            classes = pred_info.cls.tolist()
            bboxes = pred_info.xyxy.tolist()
            conf = pred_info.conf.tolist()

            # submission 파일에 저장될 내용
            prediction_string = ''
            for pred_class, conf, box in zip(classes, conf, bboxes):
                prediction_string += str(int(pred_class))+ ' ' + str(conf) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
            prediction_strings.append(prediction_string)
            file_names.append(file_name)
 
    return prediction_strings, file_names

from ultralytics import YOLO
seed = 123
torch.manual_seed(seed)

args = get_args()

# Hyperparameters etc. 
DEVICE = "cuda" if torch.cuda.is_available else "cpu" 
LOAD_MODEL_FILE = args.model # inference에 사용할 model


def inference():
    # model 생성
    model = YOLO(LOAD_MODEL_FILE) 
    # inference에 사용할 model 로드
    
    # annotation 경로
    annotation = '../dataset/test.json'
    data_dir = '../dataset/images' # dataset 경로
    
    # 데이터셋 로드
    test_dataset = CustomDataset(annotation, data_dir)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8, 
        shuffle=False,
        num_workers=4
    )  
    # 예측 및 submission 파일 생성
    prediction_strings, file_names = get_bboxes(
        annotation, test_data_loader, model
    )
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('./yolo_submission.csv', index=None)
    print(submission.head())

inference()