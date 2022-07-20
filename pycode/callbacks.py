"""
@author: Ivar Wamelink

Callback functions for model training
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import random
import io

colors = {'w': '\033[38m',
          'r': '\033[31m',
          'g': '\033[33m',
          'y': '\033[33m',
          'b': '\033[34m',
          'p': '\033[35m',
          'l': '\033[36m',
          'gray': '\033[37m'}

def save_checkpoint(uname, save_weights=True, monitor='val_loss', save_best_only=True):
    """Checkpoint save callback function.
    uname: string name of savefolder.
    save_weights: Boolean to save weights or more. Default = True
    monitor: (string) loss that is monitored for saving. Better loss equals save model. Default = 'val_loss'"""
    return tf.keras.callbacks.ModelCheckpoint(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}' + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                              save_weights_only=save_weights,
                                              monitor=monitor,
                                              save_best_only=save_best_only)

def csvlogger(uname, append=True):
    """Log training data in csv file (loss, acc, etc...)
    uname: name of savefolder.
    append: is the value appended to the csv file after training or replaced. Default = True (=append)"""
    return tf.keras.callbacks.CSVLogger('/home/iwamelink/projects/GLIOCARE/Synthetic_T1/csv/{}.csv'.format(uname), ',', append=append)

def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif epoch % 10 == 0:
        print('Learning rate decreased by: ', lr/100)
        return lr - (lr/100)
    else:
        return lr

def lr_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(scheduler)

def tensorboard(log_dir):
    """Tensorboard callback.
    log_dir: the log file where the tensorboard values are being saved. """
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

def image_callback(uname, validation_generator, model, rand=False, training_generator=None):
    """Image callback function to save the images and plot them in tensorboard.
    :param uname: name of the unet.
    :param validation_generator: the validation generator used for prediction.
    :param model: model for prediction.
    :param rand: boolean that indicates whether random or constant slices should be used.
    :param training_generator: training generator for predicting on trained data. If not specified is None.
    :return: returns a lambdacallback function that is triggered on the epoch end.
    """
    file_writer = tf.summary.create_file_writer(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/images')

    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
        The supplied figgure is closed and inaccessible after this call."""

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def plot_predictions(ground, pred):
        fig, axs = plt.subplots(1, 2)
        if rand:
            randint = random.randint(0, 95)
        else:
            randint = 100
        axs[0].imshow(pred[..., randint, 0])
        axs[0].set_title('Prediction')
        axs[1].imshow(ground[..., randint, 0])
        axs[1].set_title('Ground truth')

        return fig

    def log_images(epoch, logs):
        x, grounds = validation_generator[0] if training_generator is None else training_generator[0]
        preds = model(x)

        val_train = 'val' if training_generator is None else 'train'

        for i in range(2):
            figure = plot_predictions(grounds[i, ...], preds[i, ...])
            image = plot_to_image(figure)

            with file_writer.as_default():
                tf.summary.image('Image {} {}'.format(val_train, i), image, step=epoch + 1)

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)


class AnnealingWeight(tf.keras.callbacks.Callback):

    def __init__(self, weight, start, step):
        self.weight = weight
        self.step = step
        self.start = start

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start:
            new_weight = min(tf.keras.backend.get_value(self.weight) + (self.step / 80), 1.)
            tf.keras.backend.set_value(self.weight, new_weight)
        print(colors['p'], 'Current annealing Weight is ' + str(tf.keras.backend.get_value(self.weight)), colors['w'])
