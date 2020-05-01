"""
The neural network for imitation Learning
@author: Ehsan
"""
import tensorflow as tf

#%% Netwrok
# Building a convolutional layer
def conv_block(x, n_filters, krl_size, stride, drop_rate):
    layer = tf.keras.layers.Conv2D(n_filters, krl_size, stride, padding='same',
                                   activation='relu', use_bias=True)(x)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(drop_rate)(layer)
    return layer

# Building a fully connected layer
def fc_block(x, n_units, drop_rate):
    layer = tf.keras.layers.Dense(n_units, activation='relu', use_bias=True)(x)
    layer = tf.keras.layers.Dropout(drop_rate)(layer)
    return layer

# Building a convolutional network for images
def img_network(x):
    # parameters: n_filters, krl_size, stride, drop_rate
    network1 = conv_block(x,         32, 5, 2, 0.5) # layer 1
    network1 = conv_block(network1,  32, 3, 1, 0.5) # layer 2
    network1 = conv_block(network1,  64, 3, 2, 0.5) # layer 3
    network1 = conv_block(network1,  64, 3, 1, 0.5) # layer 4
    network1 = conv_block(network1, 128, 3, 2, 0.5) # layer 5
    network1 = conv_block(network1, 128, 3, 1, 0.5) # layer 6
    network1 = conv_block(network1, 256, 3, 1, 0.5) # layer 7
    network1 = conv_block(network1, 256, 3, 1, 0.5) # layer 8
    network1 = tf.keras.layers.Flatten()(network1)
    network1 = fc_block(network1, 512, 0.2)         # layer 9
    network1 = fc_block(network1, 512, 0.2)         # layer 10
    return network1

# Building a fully cocnnected network for measurements
def msr_network(x):
    # parameters: n_units, drop_rate
    network2 = fc_block(x,        128, 0.2) # layer 1
    network2 = fc_block(network2, 128, 0.2) # layer 2
    return network2

# Concatenating networks and building the full network
def full_network(img_input, msr_input):
    img_module = img_network(img_input)    
    msr_module = msr_network(msr_input)
    network3 = tf.keras.layers.concatenate([img_module, msr_module])
    network3 = fc_block(network3, 512, 0.2)
    network3 = fc_block(network3, 256, 0.2)
    network3 = fc_block(network3, 256, 0.2)
    network3 = tf.keras.layers.Dense(3)(network3)
    return network3
