"""
@author: Ivar Wamelink

Load data and train model.
"""
# import io
import os
# import sys
# import pickle
import matplotlib
# import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# import tensorflow.keras.losses as lss

from glob import glob
from pycode.data import DataGenerator
from keras_unet.models import custom_vnet # To find the file location of the source code, you can use Ctrl + Shift + i
from pycode.funcs import save_loc, save_model_info
from pycode.callbacks import save_checkpoint, csvlogger, lr_scheduler, tensorboard, image_callback, AnnealingWeight

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
# tf.config.experimental.set_memory_growth(devices[1], True)

colors = {'w': '\033[38m',
          'r': '\033[31m',
          'g': '\033[33m',
          'y': '\033[33m',
          'b': '\033[34m',
          'p': '\033[35m',
          'l': '\033[36m',
          'gray': '\033[37m'}

datagen_params = {'batch_size': 2,
                  'dim': (240, 240, 168),
                  'n_channels_i': 3,
                  'n_channels_o': 1}

vnet_params = {'input_shape': (240, 240, 168, datagen_params['n_channels_i']),
               'num_layers': 1,
               'filters': 2,
               'output_activation': 'linear',
               'num_classes': 1,
               'dropout': 0.5,
               # 'dropout_type': 'standard',
               # 'use_dropout_on_upsampling': True,
               # 'use_attention': True,
               }
learning_rate = 0.001

# Load training data
input = sorted(glob('/home/iwamelink/projects/IMAGO/BraTS/normalized/input/*'))
ground_truth = sorted(glob('/home/iwamelink/projects/IMAGO/BraTS/normalized/contrast/*'))

print(colors['p'], 'First training on 80% and validating on 20% of the total dataset. No test dataset used in current training!', colors['w'])
training_generator = DataGenerator(input[: round(0.8*len(input))], ground_truth[: round(0.8*len(input))], **datagen_params)
validation_generator = DataGenerator(input[round(0.8*len(input)):], ground_truth[round(0.8*len(input)):], **datagen_params)

# Create save folder
uname = save_loc()

# Save datgen and model params + save current scripts used
save_model_info(uname, vnet_params, datagen_params)

# a is the input from the validation set, b is the ground truth from the validation set; used to monitor performance in the callback
a, b = validation_generator[0]

# Execute the model over the two GPUs and perform computations on both
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# SSIM loss function
def ssim(y_true, y_pred):
    return tf.reduce_sum(1 - tf.image.ssim_multiscale(y_true, y_pred, 1, power_factors=(0.0448, 0.2856, 0.3001, 0.2363)))

# MSE loss function
def mse(y_true, y_pred):
    return tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))

# SSIM + MSE loss function
weight = tf.keras.backend.variable(0.0) # (annealing) weight is used to increase the influence of the SSIM over time.
def loss_func(y_true, y_pred):
    loss_ssim = ssim(y_true, y_pred)
    loss_mse = mse(y_true, y_pred)
    print(colors['p'], f'Current annealing weight is: {weight}', colors['w'])
    return tf.reduce_sum(loss_mse) + weight * tf.reduce_sum(loss_ssim)

# Create the 'empty' model on the GPUs with the vnet_params, optimizer, learning rate and loss function.
# with strategy.scope():
model = custom_vnet(**vnet_params)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mse', ssim])
#%% Callbacks
# cd /home/iwamelink/miniconda3/envs/dl_phd/lib/python3.8/site-packages/tensorboard
# python main.py --logdir=/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_22-06-15_1/

log_dir = f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/'
rand = False
print('{}Currently{} printing the same images (and thus is the validation set {}shuffeling){}'.format(colors['p'], 'not' if rand else '', 'not' if rand else '', colors['w']))

callbacks = [save_checkpoint(uname),
             csvlogger(uname),
             lr_scheduler(),
             tf.keras.callbacks.TerminateOnNaN(),
             tensorboard(log_dir),
             image_callback(uname, validation_generator, model, rand),      # show prediction validation vs GT
             image_callback(uname, validation_generator, model, rand, training_generator)]#,
             # AnnealingWeight(weight, 20, 1)]     # show prediction training vs GT vs prediction validation

#%%
matplotlib.use('agg')
history = model.fit(x=training_generator,
                    epochs=500,
                    validation_data=validation_generator,
                    callbacks=callbacks)

