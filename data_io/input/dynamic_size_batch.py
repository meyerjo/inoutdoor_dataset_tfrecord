from numpy import array, empty_like

from data_io.parser.boundingbox_parser import boundingbox_example_parser_batch_size_one
from data_io.parser.dynamic_boundingbox_parser import boundingbox_example_parser
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.data.util import nest

def input_fn(is_training, filenames, batch_size=1, num_epochs=1):
    """A simple input_fn using the tf.data input pipeline."""
    dataset = tf.data.TFRecordDataset(filenames)
    augmentation = False
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        parser = lambda x: boundingbox_example_parser_batch_size_one(x, augmentation)
    else:
        parser = lambda x: boundingbox_example_parser_batch_size_one(x)
    dataset = dataset.repeat(num_epochs)
    num_threads = 1 if num_epochs == 1 else 20
    dataset = dataset.map(parser, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(batch_size * 30)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=
        (
            tf.TensorShape([None, None, 3]), # input rgb
            tf.TensorShape([None, None, 3]), # input depth
            tf.TensorShape([None, 4]), # bboxes
            tf.TensorShape([None]), # labels
            tf.TensorShape([]), # number of boxes
            tf.TensorShape([]) # number of boxes
        ),
    )
    return dataset.make_one_shot_iterator().get_next()
