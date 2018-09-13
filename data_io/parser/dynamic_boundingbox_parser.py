import tensorflow as tf

from data_io.tfrecord_decoding import _recover_boundingboxes, _recover_shape

DEBUG = False

def boundingbox_example_parser(serialized_example, config, augmentation_fn=None):
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
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/depth': tf.VarLenFeature(tf.int64),
        'boxes/length': tf.VarLenFeature(tf.int64),
    }
    # parse the features
    features = tf.parse_single_example(serialized_example, feature_def)
    # reshape the image
    shape = _recover_shape(features)
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.cast(image, tf.uint8)
    shape = tf.Print(shape, [tf.shape(image)], 'Data: ')
    shape = tf.Print(shape, [features['image/format']], 'Data: ')
    image = tf.reshape(image, shape)

    # get the bounding boxes
    boundingboxes = _recover_boundingboxes(features)
    # get the number of boxes
    number_of_boxes = tf.cast(features['boxes/length'].values, tf.int32)
    number_of_boxes = tf.reduce_sum(number_of_boxes)
    # cast the labels and expand the dims so that they correspond the expectation
    labels = tf.cast(features['image/object/class/label'].values, tf.int32)
    labels = tf.expand_dims(labels, 0)

    # image shall have a shape of 4d
    # if len(image.get_shape().as_list()) == 3:
    #     image = tf.expand_dims(image, axis=0)
    # if len(boundingboxes.get_shape().as_list()) == 2:
    #     boundingboxes = tf.expand_dims(boundingboxes, axis=0)
    return image, boundingboxes, labels, tf.convert_to_tensor([[number_of_boxes]])
