"""
@author: Ivar Wamelink

Datagenerator class for general networks.
"""
import time

import numpy as np
import tensorflow as tf
import random
import nibabel as nib

class DataGenerator(tf.keras.utils.Sequence):

    """
    Returns a datagenerator. Note that this does not split the data into train, val, and test.

    Parameters
    ----------
    filenames = list of file locations.
    labels = list of labels/ground truth that correspond to the correct filenames.
    batch_size = int. Default is 32.
    dim = 2D or 3D int identification of image size. Default is (256, 256).
    n_channels_i = int. Number of input channels. Default is 3.
    n_channels_o = int. Number of output channels. Default is 1.
    shuffle = Boolean. Shuffle at the end of every epoch. Default is True.
    """

    def __init__(self,
                 filenames,
                 labels,
                 batch_size=32,
                 dim=(240, 240, 155),
                 n_channels_i=3,
                 n_channels_o=1,
                 shuffle=True):
        assert type(filenames) == list, "filenames needs to be a list."
        assert type(labels) == list, "labels needs to be a list."
        assert type(batch_size) == int, "batch_size needs to be an int"
        assert type(dim) == tuple, "dim needs to be either a 2D or 3D tuple"
        assert type(n_channels_i) == int, "n_channels_i needs to be an int"
        assert type(n_channels_o) == int, "n_channels_o needs to be an int"
        assert type(shuffle) == bool, "shuffle needs to be a boolean"


        # filenames contains all the locations of the different volumes.
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.nci = n_channels_i
        self.nco = n_channels_o
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.filenames) / self.batch_size))

    # index is the batch number of the current batch.
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generates a batch with batch size and filenames.
        indexes_i = self.filenames[index * self.batch_size: (index + 1) * self.batch_size]
        indexes_o = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

        return self.__data_generation(indexes_i, indexes_o)

    def on_epoch_end(self):
        """Updates indexes after each epoch; shuffeling the input order"""
        if self.shuffle:
            shuffle = list(zip(self.filenames, self.labels))
            random.shuffle(shuffle)

            self.filenames, self.labels = zip(*shuffle)

    def __data_generation(self, filenames_temp, labels_temp):
        """Generates batch"""
        x = np.zeros((self.batch_size, *self.dim, self.nci))
        y = np.zeros((self.batch_size, *self.dim, self.nco))

        for i, volume in enumerate(filenames_temp):
            try:
                raw_volume = nib.load(volume).get_fdata()
                # Slice normalization
                x[i, ..., 7:162, :] = raw_volume
                del raw_volume

                raw_volume = nib.load(labels_temp[i]).get_fdata()
                y[i, ..., 7:162, 0] = raw_volume

            except Exception as e:
                print(e)

        return x.astype(np.float), y.astype(np.float)
