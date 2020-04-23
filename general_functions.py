"""
General functions to be used in imitation learning project
@author: Ehsan
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#%% General Functions
# making a list of files with a specific extension in a directory
def filelist(path, ext):
    from os import listdir
    file = [f for f in listdir(path) if f.endswith(ext)]
    file.sort()
    return file

# Visualize the original and augmented images for comoparison
def visualize(original, augmented):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.subplot(1,2,2)
    plt.title('Augmented Image')
    plt.imshow(augmented)

# Normalizing values of a data frame
# cat_columns is a list of categorical columns.
def normalize(df,cat_columns):
    df_cat = pd.DataFrame()
    for i in cat_columns:
        df_cat = pd.concat([df_cat, df.pop(i)], axis=1)
    df = (df - df.mean()) / df.std()
    df = pd.concat([df_cat, df], axis=1)
    return df

# Reading a signel image
def read_image(file, des_img_size):
    img_string = tf.io.read_file(file)
    image = tf.io.decode_image(img_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, des_img_size, preserve_aspect_ratio=True,
                            method=tf.image.ResizeMethod.BILINEAR)
    return image
