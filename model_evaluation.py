"""
Evaluating the Imitation Learning Model
@author: Ehsan
"""
import pandas as pd
import tensorflow as tf
import os
from general_functions import filelist

model_path = r'...\whole_model.h5'
measure_path = r'...\All_measurements_truncated.csv'
image_dir = r'...\images'
des_img_size = [88,200]
cat_columns = ['throttle_fl', 'brake_fl']

def augment(image):
    image = tf.image.convert_image_dtype(image, tf.float32)               # converting and scaling to [0,1]
    image = tf.image.resize(image, des_img_size, preserve_aspect_ratio=True,
                            method=tf.image.ResizeMethod.BILINEAR)        # resizing
    return image

def normalize(df,cat_columns):
    df_cat = pd.DataFrame()
    for i in cat_columns:
        df_cat = pd.concat([df_cat, df.pop(i)], axis=1)
    df = (df - df.mean()) / df.std()
    df = pd.concat([df_cat, df], axis=1)
    return df

# reading images
input_images = []
files = filelist(image_dir, ext='.png')
for k in files:
    img_string=tf.io.read_file(os.path.join(image_dir,k))
    image=tf.io.decode_image(img_string, channels=3)
    augm_img = augment(image)
    input_images.append(augm_img)

# reading measurements
df_measure = pd.read_csv(measure_path, index_col=None, header='infer')
df_measure = normalize(df_measure,cat_columns)
for i in cat_columns:
    df_measure[i] = pd.Categorical(df_measure[i])
control_output = df_measure.iloc[:,0:3]
control_output = tf.convert_to_tensor(control_output.values, dtype=tf.float32)
input_measure = df_measure.iloc[:,3:]
input_measure = tf.convert_to_tensor(input_measure.values, dtype=tf.float32)

# loading the model
model = tf.keras.models.load_model(model_path)

# evaluating the model with current data
loss, acc = model.evaluate([input_images, input_measure], control_output, verbose=1)
