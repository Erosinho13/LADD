import torch
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
import collections
from PIL import Image
import warnings
import math
from PIL import ImageFilter

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl=None):
        if lbl is not None:
            for t in self.transforms:
                img, lbl = t(img, lbl)
            return img, lbl
        else:
            for t in self.transforms:
                img = t(img)
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class FixedResize(object):
    def __init__(self, width=None, height=None,  interpolation=Image.BILINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img, lbl=None):
        old_width, old_height = img.size
        if self.height is None and self.width is not None:
            new_height = round(self.width * old_height / old_width)
            size = (new_height, self.width)
        elif self.height is not None and self.width is None:
            new_width = round(self.height * old_width / old_height)
            size = (self.height, new_width)
        elif None not in [self.width, self.height]:
            size = (self.height, self.width)

        if lbl is not None:
            return F.resize(img, size, self.interpolation), F.resize(lbl, size, Image.NEAREST)
        else:
            return F.resize(img, size, self.interpolation)


class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl=None):
        if lbl is not None:
            return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)
        else:
            return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img, lbl=None):

        if lbl is None:
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

            return F.center_crop(img, self.size)

        else:
            assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
                lbl = F.pad(lbl, (self.size[1] - lbl.size[0], 0), 255, self.padding_mode)

            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
                lbl = F.pad(lbl, (0, self.size[0] - lbl.size[1]), 255, self.padding_mode)

            return F.center_crop(img, self.size), F.center_crop(lbl, self.size)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl=None):
        if lbl is not None:
            return F.center_crop(img, self.size), F.center_crop(lbl, self.size)
        else:
            return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'


class Pad(object):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, lbl=None):
        if lbl is not None:
            return F.pad(img, self.padding, self.fill, self.padding_mode), F.pad(lbl, self.padding, self.fill,
                                                                                 self.padding_mode)
        else:
            return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class Lambda(object):

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img, lbl=None):
        if lbl is not None:
            return self.lambd(img), self.lambd(lbl)
        else:
            return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        angle = self.get_params(self.degrees)
        if lbl is not None:
            return F.rotate(img, angle, self.resample, self.expand, self.center), \
                   F.rotate(lbl, angle, self.resample, self.expand, self.center)
        else:
            return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl=None):
        if random.random() < self.p:
            if lbl is not None:
                return F.hflip(img), F.hflip(lbl)
            else:
                return F.hflip(img)
        if lbl is not None:
            return img, lbl
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            if lbl is not None:
                return F.vflip(img), F.vflip(lbl)
            else:
                return F.vflip(img)
        if lbl is not None:
            return img, lbl
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomScale(object):

    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl=None):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.size[1] * scale), int(img.size[0] * scale))

        if lbl is not None:
            if img.size != lbl.size:
                img = F.resize(img, (lbl.size[1], lbl.size[0]), self.interpolation)
            return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)
        else:
            return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(scale_range={0}, interpolation={1})'.format(self.scale_range,
                                                                                       interpolate_str)


class ToTensor(object):

    def __call__(self, pic, lbl=None):
        if lbl is not None:
            return F.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=np.uint8))
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl=None):
        if lbl is not None:
            return F.normalize(tensor, self.mean, self.std), lbl
        else:
            return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomCrop(object):

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl=None):
        if lbl is None:
            if self.padding > 0:
                img = F.pad(img, self.padding)
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))

            i, j, h, w = self.get_params(img, self.size)

            return F.crop(img, i, j, h, w)

        else:
            assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
            if self.padding > 0:
                img = F.pad(img, self.padding)
                lbl = F.pad(lbl, self.padding)

            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
                lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2), fill=255)

            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
                lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2), fill=255)

            i, j, h, w = self.get_params(img, self.size)

            return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomResizedCrop(object):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, lbl=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if lbl is not None:
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
                   F.resized_crop(lbl, i, j, h, w, self.size, Image.NEAREST)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl=None):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        if lbl is not None:
            return transform(img), lbl
        else:
            return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomScaleRandomCrop(object):

    def __init__(self, crop_size=(1024, 2048), scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)):
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self, img, lbl=None):

        scale = self.scale[(torch.ones(len(self.scale)) / len(self.scale)).multinomial(num_samples=1)]

        if scale < 1:
            tr = Compose([
                RandomScale((scale, scale)),
                PadCenterCrop((self.crop_size[0], self.crop_size[1]), fill=255)
            ])
        else:
            tr = Compose([
                RandomScale((scale, scale)),
                RandomCrop(self.crop_size)
            ])

        if lbl is not None:
            return tr(img, lbl)

        return tr(img)


class GaussianBlur(object):

    def __call__(self, img, lbl=None):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        if lbl is not None:
            return img, lbl
        else:
            return img
