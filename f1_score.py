import tensorflow as tf
from keras.applications import VGG16
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 이미지 데이터셋 로드 및 전처리
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './test_preprocess/',
    image_size=(224, 224),
    batch_size=64,
    subset=None,
    seed=123
)

def preprocessing(i, score):
    i = tf.cast(i/255.0, tf.float32)
    return i, score

test_ds = test_ds.map(preprocessing)

test_images = []
true_labels = []

for images, labels in test_ds:
    test_images.append(images.numpy())
    true_labels.append(labels.numpy())

test_images = np.concatenate(test_images, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# 모델 로드 및 컴파일
loaded_model = tf.keras.models.load_model('./model_folder/test_model6')
loaded_model.summary()

loaded_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
    metrics=["accuracy"]
)

# 모델 예측
predictions = loaded_model.predict(test_images)

# F1 스코어 계산
f1 = f1_score(true_labels, predictions.round())

print("F1 Score:", f1)