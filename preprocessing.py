from keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './pre/',
    image_size=(224,224),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './pre/',
    image_size=(224,224),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=123
)


"""
true_image = np.concatenate([X.numpy() for _, X in val_ds], axis=0)
true_labels = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
print(true_image) """


def Preprocessing2(i, score):
    i = tf.cast(i/255.0, tf.float32)
    return i, score


train_ds = train_ds.map(Preprocessing2)
val_ds = val_ds.map(Preprocessing2)



model = tf.keras.models.Sequential([
    #tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal' , input_shape=(64,64,3)),
    #tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    #tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D( 32, (3,3), padding="same", activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Conv2D( 64, (3,3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D( 128, (3,3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),

])


model.summary()

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('test_model6' + str(int(time.time()))) )
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, mode='max')


model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
    metrics=["accuracy"]
)



model.fit(train_ds, validation_data=(val_ds), epochs =15, callbacks=[tensorboard,es] )

model.save('./model_folder/test_model6')