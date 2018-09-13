import tensorflow as tf


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_boundingbox_tfrecord(boundingbox_structs, tfrecord_filename, load_fn=None):
    """ Creates a .tfrecord file for object detection, compatibile with
        the tensorflow detection API.

    Args:
        boundingbox_structs: A list of `data.BoundingBoxStructs`.
        tfrecord_filename: `string`, the filepath used to store the tfrecord.
    """
    index_written_file = 0
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        for i, box_struct in enumerate(boundingbox_structs):
            if i % 100 == 0:
                print('Writing boxes for file {0}/{1}'.format(i, len(boundingbox_structs)))
            # load function
            image_data_rgb = tf.gfile.FastGFile(box_struct.img_path_rgb, 'r').read()
            image_data_depth = tf.gfile.FastGFile(box_struct.img_path_depth, 'r').read()


            file_name = box_struct.img_path_rgb.encode('utf-8')

            if box_struct.file_format is not None:
                img_format = _bytes_feature(box_struct.file_format.encode('utf-8'))
            else:
                img_format = _bytes_feature('depth'.encode('utf-8'))
            #
            normalization_factor_x = 1.
            normalization_factor_y = 1.
            #
            xmin_values = [box.xmin/normalization_factor_x for box in box_struct.boundingboxes]
            xmax_values = [box.xmax/normalization_factor_x for box in box_struct.boundingboxes]
            ymin_values = [box.ymin/normalization_factor_y for box in box_struct.boundingboxes]
            ymax_values = [box.ymax/normalization_factor_y for box in box_struct.boundingboxes]

            label_values = [box.label for box in box_struct.boundingboxes]
            label_text = ['person' for box in box_struct.boundingboxes]
            # make sure that boxes and labels have the same length
            if not (len(xmin_values) == len(xmax_values) == len(ymin_values) == len(ymax_values) == len(label_values)):
                raise ValueError('All boxes must have the same size')

            feat = {
                'image/height': _int64_feature(box_struct.height),
                'image/width': _int64_feature(box_struct.width),
                'image/object/bbox/xmin': _float_feature(xmin_values),
                'image/object/bbox/xmax': _float_feature(xmax_values),
                'image/object/bbox/ymin': _float_feature(ymin_values),
                'image/object/bbox/ymax': _float_feature(ymax_values),
                'image/object/class/label': _int64_feature(label_values),
                'image/object/class/text': _bytes_feature(label_text),
                'image/rgb/encoded': _bytes_feature(image_data_rgb),
                'image/depth/encoded': _bytes_feature(image_data_depth),
                'image/format': img_format,
                'image/filename': _bytes_feature(file_name),
                'image/depth': _int64_feature([3]),
                'boxes/length': _int64_feature([len(box_struct.boundingboxes)]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())
            index_written_file += 1
        writer.close()
    print('Written examples: {0}'.format(index_written_file))
