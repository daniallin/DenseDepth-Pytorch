import random
import numpy as np
import torch

import scipy.ndimage as ndimage
from PIL import Image


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w, :]

        landmarks = landmarks[top: top + new_h,
                    left: left + new_w]

        return image, landmarks


class CropCenter(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        inputs = inputs[y1: y1 + th, x1: x1 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs, target


class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation, cannot be more than 180 degrees
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, angle, diff_angle=0, order=2):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, sample):
        inputs, target = sample
        h, w, _ = inputs.shape

        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2

        angle1_rad = angle1 * np.pi / 180

        inputs = ndimage.interpolation.rotate(inputs, angle1, reshape=True, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=True, order=self.order)

        # keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi / 2 - np.abs(angle1_rad % np.pi - np.pi / 2)

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c_diag = h / np.sqrt(h * h + w * w)
        s_diag = w / np.sqrt(h * h + w * w)

        ratio = 1. / (c1 + w / float(h) * s1)

        crop = CropCenter((int(h * ratio), int(w * ratio)))
        return crop(inputs, target)


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample
        if np.random.randn() > 0.5:
            image = image[:, ::-1, :]
            landmarks = landmarks[:, ::-1]

        return image, landmarks


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sample):
        image, landmarks= sample

        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        return image, landmarks

