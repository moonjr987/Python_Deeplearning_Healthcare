import os
import json
import numpy as np
import glob
import shutil
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from PIL import Image


# 이미지 전처리 및 라벨 생성하는 소중한 함수
def preprocessing(image_data, json_data):
    images = []  # 이미지 데이터 모셔두는 곳
    labels = []  # json 데이터 모셔두는 곳
    for i, json_path in enumerate(os.listdir(json_data), start=1):
        if json_path.endswith('.json'):
            with open(os.path.join(json_data, json_path), 'r') as json_file:
                json_str = json_file.read()
                json_str_fixed = json_str.replace('false', 'False').replace('true', 'True')
                data = eval(json_str_fixed)
            if any(tooth['decayed'] for tooth in data['tooth']) == True:
                try:
                    name = data['image_filepath'].split('/')[-1]
                    image_path = os.path.join(image_data, name)
                    image_path = image_path.replace('\\', '/')
                    print(image_path)
                    shutil.copyfile(image_path, './test_preprocess/true/'+ name)


                except FileNotFoundError:
                    print(f"File not found: {image_path}")
                    continue

            else:
                try:
                    name = data['image_filepath'].split('/')[-1]
                    image_path = os.path.join(image_data, name)
                    image_path = image_path.replace('\\', '/')
                    shutil.copyfile(image_path, './test_preprocess/false/'+ name)
                except FileNotFoundError:
                    print(f"File not found: {image_path}")
                    continue







image_data = './Dataset/test_data/image'  # 이미지 경로
json_data = './Dataset/test_data/json/'  # json 경로

preprocessing(image_data, json_data)