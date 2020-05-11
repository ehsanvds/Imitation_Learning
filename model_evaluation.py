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
