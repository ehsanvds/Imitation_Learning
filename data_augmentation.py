"""
Imitation Learning for Following a Car

@author: Ehsan
March 25, 2020
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#%% Image Augmentation
def gaussian_blur(image, kernel_size, sigma=3):
    ax = tf.range(-kernel_size//2+1, kernel_size//2+1)
    x, y = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(x**2+y**2)/(2*sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, tf.shape(image)[-1]])
    gaussian_kernel = tf.dtypes.cast(kernel[..., tf.newaxis],tf.float32)
    return tf.nn.depthwise_conv2d(image[tf.newaxis, ...], gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')

def gaussian_noise(image, noise_lev):
    noise = tf.random.normal(tf.shape(image), mean=0, stddev=noise_lev)
    noise_img = image + noise
    noise_img = tf.clip_by_value(noise_img, 0, 1)
    return noise_img

def img_dropout(image, max_num_reg):
    image = image[tf.newaxis, ...]
    for i in range(np.random.randint(max_num_reg+1)):
        cords = np.random.random(size=3)
        box = [cords[0], cords[1], cords[2],
               np.random.rand()/(100*(cords[2]-cords[0]))+cords[1]]
        boxes = tf.dtypes.cast(box,tf.float32)[tf.newaxis,tf.newaxis,...]
        image = tf.image.draw_bounding_boxes(image, boxes, colors=[[0,0,0,1]])
    return image

def augment(data):
    image = data['img_input']
    # image = tf.squeeze(image, axis=[0])
    image = tf.image.random_contrast(image, 0.8, 1.2)                     # changing contrast
    image = tf.image.random_brightness(image, 0.2)                        # changing brightness
    image = tf.image.random_saturation(image, 0.8, 1.2)                   # changing saturation
    image = gaussian_blur(image, np.random.randint(1,4))[0]               # adding Gaussian blur
    image = gaussian_noise(image, noise_lev=np.random.randint(0,40)/1000) # adding Gaussian noise
    image = img_dropout(image, max_num_reg=4)[0]                          # region dropout
    return {'img_input':image, 'msr_input':data['msr_input']}

