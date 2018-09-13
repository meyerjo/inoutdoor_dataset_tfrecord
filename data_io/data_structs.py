# Copyright team frAIburg, 2017
# author: Philipp Jund (jundp@informatik.uni-freiburg.de)
from __future__ import division

import numpy as np
import tensorflow as tf
from PIL import Image
import os


def normalize_img(img):
    """ Converts an np.array with dtype=np.uint8 to float in [-0.5, 0.5] """
    assert isinstance(img, np.ndarray)
    assert(img.dtype == np.uint8)
    return img / 255 - 0.5


def load_image(path, normalize=False):
    """ Loads an image, uint8 if normalize=False else float in [-0.5, 0.5] """
    img = np.asarray(Image.open(path), dtype=np.uint8)
    return normalize_img(img) if normalize else img


class BoundingBox:
    """ A BoundingBox, consisting of coordinates and a label. """
    __slots__ = ("xmin", "xmax", "ymin", "ymax", "label", "label_str", "valid",
                 "normalized")

    def __init__(self, xmin, xmax, ymin, ymax, label, label_str='', normalized=True):
        self.normalized = normalized
        self.xmin = self._assert_normalized(xmin)
        self.xmax = self._assert_normalized(xmax)
        self.ymin = self._assert_normalized(ymin)
        self.ymax = self._assert_normalized(ymax)
        self.valid = self.validate_box()
        self.label = label
        self.label_str = label_str

    def _assert_normalized(self, x):
        """ Asserts that x is normalized and returns x. """
        if not self.normalized:
            return x
        assert (x >= 0 and x < 1), "Box coordinates must be normalized!"
        return x

    def validate_box(self):
        """ Checks that height and width are not too small, and that
            max on axis is larger than min on axis
        """
        assert self.xmax >= self.xmin, "xmax should be larger than xmin!"
        assert self.ymax >= self.ymin, "ymax should be larger than ymin!"
        if self.xmax - self.xmin < 0.005:
            print("Warning: Invalid box, width is too small.")
            return False
        if self.ymax - self.ymin < 0.005:
            print("Warning: Invalid box, height is too small.")
            return False
        return True


class BoundingBoxStruct:
    """ Struct to tie multiple bounding boxes and an image path.
        It serves as an abstraction to deal with several different annotation
        formats.
    """

    __slots__ = ("img_path_rgb", "img_path_depth", "boundingboxes", "file_format", "height", "width")

    def __init__(self, img_path_rgb, img_path_depth, boundingboxes, height=-1, width=-1):
        self.img_path_rgb = img_path_rgb
        self.img_path_depth = img_path_depth
        self.boundingboxes = boundingboxes
        self.file_format = self._get_format()
        if width == -1 or height == -1:
            height, width = self.load('ppm').shape[:2]
            height = int(height)
            width = int(width)
        self.width = width
        self.height = height

    def load(self, encoding="numpy"):
        """ Loads the image.

        Args:
            encoding: Either "numpy" or "jpg/png"
        """
        if not self.img_path_rgb:
            raise ValueError("No image path specified.")
        if not os.path.exists(self.img_path_rgb):
            raise ValueError("No image in this path")
        if encoding == "numpy" or encoding == "ppm":
            return load_image(self.img_path_rgb)
        if encoding == "jpg/png":
            return tf.gfile.FastGFile(self.img_path_rgb, 'rb').read()
        else:
            raise ValueError("Unknown encoding. Must be one"
                             "of {'numpy', 'jpg/png'} " + str(encoding))

    def get_pixel_boundingboxes(self):
        return [BoundingBox(b.xmin * self.width, b.xmax * self.width,
                            b.ymin * self.height, b.ymax * self.height,
                            b.label, False) for b in self.boundingboxes]

    def _get_format(self):
        """ Determines the file format from the path.
            Only jpg/png are supported.
        """
        p = self.img_path_rgb
        if p.endswith(".png") or p.endswith(".PNG"):
            return ".png"
        if (p.endswith(".jpg") or p.endswith(".JPG") or
                p.endswith(".jpeg") or p.endswith(".JPEG")):
            return ".jpeg"
        if (p.endswith(".ppm")):
            return ".ppm"
