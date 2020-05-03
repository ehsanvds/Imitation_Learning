"""
Imitation Learning
@author: Ehsan
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os

#%% set parameters (check the input path)
measure_path = r'...\All_measurements_truncated.csv'
image_dir = r'...\images'
ext = '.png'
n_msr_param = 4 # number of parameters for the measurements
cat_columns = ['throttle_fl', 'brake_fl'] # categorical columns in the measurements

#%% Main
if __name__=='__main__':
    # reading the images
    print('Reading images ...')
    input_images = []
    files = filelist(image_dir, ext)
    for k in files:
        input_images.append(read_image(os.path.join(image_dir,k)))    
    input_images = tf.convert_to_tensor(input_images, dtype=tf.float32)
    # visualize(image, augm_img)
    
    # reading the measurements
    print('Reading measurements ...')
    df_measure = pd.read_csv(measure_path, index_col=None, header='infer')
    df_measure = normalize(df_measure,cat_columns)
    for i in cat_columns:
        df_measure[i] = pd.Categorical(df_measure[i])
    control_output = df_measure.iloc[:,0:3]
    control_output = tf.convert_to_tensor(control_output.values, dtype=tf.float32)
    input_measure = df_measure.iloc[:,3:]
    input_measure = tf.convert_to_tensor(input_measure.values, dtype=tf.float32)
    
    # building the model
    print('Building the model ...')
    img_input = tf.keras.Input(shape=tuple(np.array(tf.shape(input_images[0]))), name='img_input')
    msr_input = tf.keras.Input(shape=(n_msr_param,), name='msr_input')
    model = full_network(img_input, msr_input)
    model = tf.keras.Model(inputs=[img_input, msr_input], outputs=model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss='mean_squared_error', metrics=['accuracy'])
    print(model.summary())
