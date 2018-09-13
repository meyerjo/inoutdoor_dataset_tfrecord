from os.path import join

import os
import numpy as np
from PIL import Image

from utils.colorize_depth_map import colorize_1d_depth_map

if __name__ == '__main__':
    input_path = '/home/meyerjo/dataset/inoutdoorpeoplergbd/Depth/'

    output_path = '/home/meyerjo/dataset/inoutdoorpeoplergbd/DepthJet/'

    files = [f for f in os.listdir(input_path) if os.path.isfile(join(input_path, f))]

    for file_nr, f in enumerate(files):
        im = Image.open(join(input_path, f))

        im_np = np.asarray(im)
        im_np = im_np / float(np.max(im_np))
        im_np = im_np * 255.

        color_im_np = colorize_1d_depth_map(im_np)
        im_color = Image.fromarray(color_im_np)
        im_color.save(join(output_path, f))

        if file_nr % 100 == 0:
            print('File {0}/{1}'.format(file_nr, len(files)))
