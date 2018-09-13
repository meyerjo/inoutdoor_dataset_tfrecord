import numpy as np
from matplotlib import cm


def colorize_1d_depth_map(X, coloring_mode='jet'):
    """
    Function colorizes a depth map given a certain function
    :param X: ndarray descibring the 1-D depth image, which is already converted to a depth map 0-255
    :param coloring_mode: descirbes the default mode how to colorize a 1-D depth point, default: jet
    :return: a 3d ndarray representing the colorized depth map
    """
    assert(isinstance(X, np.ndarray))
    assert(np.max(X) <= 255.)
    if coloring_mode == 'jet':
        coloring_function = lambda x: np.delete(cm.jet(x), 3, 2)
    else:
        raise ValueError('Unknown coloring mode')
    (height, width) = (X.shape[0], X.shape[1])
    # convert it to the 0-1 range
    X = X / 255.
    # apply the color map
    C = coloring_function(X)
    # convert back to 0-255 range
    C = np.multiply(C, 255)
    return C.astype('uint8')
