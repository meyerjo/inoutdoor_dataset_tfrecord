from PIL import Image
import os
import numpy as np


def read_pgm_to_1d_depth(file_name, display=False):
    """
    Reads a pgm file to a depth map (assumes a 16-bit depth map, which is mapped to 8-bit)
    :param file_name: filename
    :param display: shall the image be displayed after reading the depth map
    :return: numpy array in the shape of the image, with one dimension in the range 0-255
    """
    assert(os.path.exists(file_name))
    M = Image.open(file_name)
    # extract the shape
    (height, width) = (M.size[0], M.size[1])
    # extract the data and convert to a numpy array
    X = np.asarray(M.getdata(), dtype=np.uint16).reshape(width, height)
    # map the depth data
    X = (X / 65535.) * 255
    if display:
        im2 = Image.fromarray(X)
        im2.show()
    return X
