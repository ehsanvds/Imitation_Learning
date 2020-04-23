"""
Imitation Learning for Following a Car

@author: Ehsan
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os



#%% Other Functions
def filelist(path, ext):
    from os import listdir
    file = [f for f in listdir(path) if f.endswith(ext)]
    file.sort()
    return file

def visualize(original, augmented):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.subplot(1,2,2)
    plt.title('Augmented Image')
    plt.imshow(augmented)

def normalize(df,cat_columns):
    df_cat = pd.DataFrame()
    for i in cat_columns:
        df_cat = pd.concat([df_cat, df.pop(i)], axis=1)
    df = (df - df.mean()) / df.std()
    df = pd.concat([df_cat, df], axis=1)
    return df

def read_image(file):
    img_string = tf.io.read_file(file)
    image = tf.io.decode_image(img_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, des_img_size, preserve_aspect_ratio=True,
                            method=tf.image.ResizeMethod.BILINEAR)
    return image