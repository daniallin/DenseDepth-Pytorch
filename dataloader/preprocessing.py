import random
import numpy as np
import torch

import scipy.ndimage as ndimage
from PIL import Image
import torch.nn.functional as F


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

        inputs = F.rotate(inputs, angle1)
        target = F.rotate(target, angle1)

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
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            landmarks = landmarks.transpose(Image.FLIP_LEFT_RIGHT)

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


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        sample (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, sample):
        image, depth = sample
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)

        return image, depth

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.float().div(255)
            else:
                return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        if nchannel == 3:
            # use .view() because PIL get image as W*H
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if nchannel == 1:
            img = img.view(pic.size[1], pic.size[0])
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class RandomSizedCrop(object):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                w = min(h, w)
                return i, j, w, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        img, depth = sample
        # print('size:{}/{}'.format(img.size[0], img.size[1]))
        ori_size = (img.size[0], img.size[1])
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # will the order of crop and resize influences the results?
        img = img.crop((j, i, j+w, i+h))
        img = img.resize(ori_size, self.interpolation)
        depth = depth.crop((j, i, j+w, i+h))
        depth = depth.resize(ori_size, self.interpolation)
        return img, depth



