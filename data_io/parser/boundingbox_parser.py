import tensorflow as tf

from data_io.tfrecord_decoding import _recover_shape, _recover_boundingboxes

DEBUG = False

def boundingbox_example_parser_batch_size_one(serialized_example, augmentation_fn=None):
    """ Parses a single tf.Example into image and label tensors. For
        compatibility reasons, this function expects the same example
        definition as the tensorflow object detection API.

    Returns:
        image: `tf.Tensor` with shape (3, input_height, input_width), tf.uint8
        boundingboxes: `tf.Tensor` with shape (4, num_boxes) as
            [ymin, xmin, ymax, xmax].
        labels: `tf.Tensor` with shape (num_boxes,), the labels for the boxes.
    """
    feature_def = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/rgb/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/depth/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/depth': tf.VarLenFeature(tf.int64),
        'boxes/length': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(serialized_example, feature_def)
    image_shape = _recover_shape(features) # assuming same size for both images

    #image_rgb = tf.decode_raw(features['image/rgb/encoded'], tf.uint8)
    image_rgb = tf.image.decode_png(features['image/rgb/encoded'], channels=3, dtype=tf.uint8)
    # image_rgb = tf.cast(image_rgb, tf.uint8)
    # image_rgb = tf.reshape(image_rgb, image_shape)

    # image_depth = tf.decode_raw(features['image/depth/encoded'], tf.uint8)
    image_depth = tf.image.decode_png(features['image/depth/encoded'], channels=3, dtype=tf.uint8)
    # image_depth = tf.cast(image_depth, tf.uint8)
    # image_depth = tf.reshape(image_depth, image_shape)

    if DEBUG: print(image_rgb.get_shape())
    if DEBUG: print(type(image_rgb), type(image_rgb))

    boundingboxes = _recover_boundingboxes(features)

    number_of_boxes = tf.cast(features['boxes/length'].values, tf.int32)
    number_of_boxes = tf.reduce_sum(number_of_boxes)

    boundingboxes = tf.reshape(boundingboxes, tf.stack([number_of_boxes, 4]))

    labels = tf.cast(features['image/object/class/label'].values, tf.int32)
    labels = tf.reshape(labels, [number_of_boxes])

    filename = tf.cast(features['image/filename'], tf.string)
    #if augmentation_fn:
    #    return augmentation_fn(image, boundingboxes, labels, num_boxes)
    return image_rgb, image_depth, boundingboxes, labels, number_of_boxes, filename