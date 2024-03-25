import os
import json
import numpy as np
import glob
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# 이미지 전처리 및 라벨 생성하는 소중한 함수
def preprocessing(image_data, json_data, preprocessing_folder='./pre/'):
    images = []  # 이미지 데이터 모셔두는 곳
    labels = []  # json 데이터 모셔두는 곳
    for i, json_path in enumerate(os.listdir(json_data), start=1):
        if json_path.endswith('.json'):
            with open(os.path.join(json_data, json_path), 'r') as json_file:
                json_str = json_file.read()
                json_str_fixed = json_str.replace('false', 'False').replace('true', 'True')
                data = eval(json_str_fixed)
            
            if any(tooth['decayed'] for tooth in data['tooth']):
                label = 1
            else:
                label = 0
            
            try:
                name = data['image_filepath'].split('/')[-1]
                image_path = os.path.join(image_data, name)
                image_path = image_path.replace('\\', '/')
                shutil.copyfile(image_path, os.path.join(preprocessing_folder, str(label), name))

            except FileNotFoundError:
                print(f"File not found: {image_path}")
                continue

# 데이터 전처리
image_data = './Dataset/test_data/image'  # 이미지 경로
json_data = './Dataset/test_data/json/'  # json 경로
preprocessing_folder = './pre/'

preprocessing(image_data, json_data, preprocessing_folder)

# 이미지 데이터셋 생성
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    preprocessing_folder,
    image_size=(224, 224),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    preprocessing_folder,
    image_size=(224, 224),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=123
)

# 이미지 데이터 전처리 함수
def Preprocessing2(i, score):
    i = tf.cast(i / 255.0, tf.float32)
    return i, score

# 이미지 데이터 전처리
train_ds = train_ds.map(Preprocessing2)
val_ds = val_ds.map(Preprocessing2)

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

# 모델 컴파일 및 훈련
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('test_model6' + str(int(time.time()))))
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, mode='max')

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard, es])

# 모델 저장
model.save('./model_folder/test_model6')