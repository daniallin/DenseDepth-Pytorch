import numpy as np
import torch
import random
import tensorflow as tf


def DepthNorm(depth, max_depth):
    return max_depth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def tf2torch_trans(v):
    # tensorflow weights to pytorch weights
    if v.shape.ndims == 4:
        v = np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.shape.ndims == 2:
        v = np.ascontiguousarray(v.transpose())
    v = torch.Tensor(v)
    return v


def keras2torch_weights(pretrained_model):
    custom_objects = {'BilinearUpSampling2D': tf.keras.layers.UpSampling2D, 'depth_loss_function': None}
    model = tf.keras.models.load_model(pretrained_model, custom_objects=custom_objects)
    keras_weights = dict()
    for layer in model.layers:
        if type(layer) is tf.keras.layers.Conv2D:
            keras_weights[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))

    return keras_weights


if __name__ == '__main__':
    weights = keras2torch_weights('../models/pretrained/nyu.h5')

    print(weights)
