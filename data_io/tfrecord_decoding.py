import tensorflow as tf
import numpy as np

from data_io.scope_wrapper import scope_wrapper


@scope_wrapper
def _serialized_img_to_tensor(features):
    """ Converts a serialized image to an image tensor of a fixed size. """
    shape = _recover_shape(features)
    #img = tf.coner(features['image/encoded'])
    #img = tf.image.decode_image(serialized_img)
    serialized_img = features['image/encoded']
    img = np.fromstring(serialized_img)
    img = serialized_img
    img = tf.cast(img, tf.uint8)
    img = tf.reshape(img, shape)
    return img


@scope_wrapper
def _recover_shape(features):
    """ Recreates the shape tensor from tf.train.Features. """
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.cast(tf.reduce_sum(features['image/depth'].values), tf.int32)
    return tf.stack((height, width, depth))


@scope_wrapper
def _recover_boundingboxes(features):
    """ Creates a list of boxes [(ymin,xmin, ...), ...] from features. """
    ymin = features['image/object/bbox/ymin'].values
    xmin = features['image/object/bbox/xmin'].values
    ymax = features['image/object/bbox/ymax'].values
    xmax = features['image/object/bbox/xmax'].values
    return tf.transpose([ymin, xmin, ymax, xmax])


def repeat(a, repeats, num_repeats=None):
    """ Repeats each element a[i] repeats[i] times. The shape of repeats must
        be known at compile time. If tensorflow is not able to infer the shape
        of repeats but it is known, num_repeats can be passed.
        E.g. a = [4, 5], repeats = [1, 3] --> [4, 5, 5, 5]
    """
    a = tf.convert_to_tensor(a)
    repeats = tf.convert_to_tensor(repeats)
    if num_repeats is None and repeats.get_shape().as_list()[0] is not None:
        num_repeats = repeats.get_shape().as_list()[0]
    if num_repeats is None:
        raise ValueError("num_repeats could not be inferred, id possible"
                         "specify it manually")
    repeated = [tf.tile([a[i]], [repeats[i]]) for i in range(num_repeats)]
    return tf.concat(repeated, axis=0)


def repeat_v2(a, number_of_boxes, total_boxes):
    with tf.control_dependencies([tf.assert_equal(tf.size(a), tf.size(number_of_boxes))]):
        max_number_of_boxes_index = tf.argmax(number_of_boxes)
        max_number_of_boxes = number_of_boxes[max_number_of_boxes_index]
        def tiling(x):
            # part_a = tf.tile([a[x]], [number_of_boxes[x]])
            part_a = tf.tile([a[x]], [number_of_boxes[x]])
            part_b = tf.tile([-1], [max_number_of_boxes - number_of_boxes[x]])
            return tf.concat([part_a, part_b], axis=0) # we have to do this to preserve the shape in all dimensions

        ids = tf.range(0, tf.size(a))
        box_ind = tf.map_fn(tiling, ids) # apply the function to all image_ids items
        box_ind = tf.reshape(box_ind, shape=(total_boxes, 1)) # reshape to vector form
        valid_box_indices = tf.greater_equal(box_ind, 0) # get a boolean mask
        box_ind = tf.boolean_mask(box_ind, valid_box_indices)
        return box_ind
