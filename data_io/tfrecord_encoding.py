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
        sess.run(tf.initialize_all_variables())
        for i, box_struct in enumerate(boundingbox_structs):
            if i % 100 == 0:
                print('Writing boxes for file {0}/{1}'.format(i, len(boundingbox_structs)))
            # load function
            if not load_fn:
                img = box_struct.load(encoding="ppm")
            else:
                orig_img, img = load_fn(box_struct.img_path)
            image_raw = img.tostring()
            image_raw = tf.image.encode_png(img)
            image_raw = sess.run(image_raw)

            if box_struct.file_format is not None:
                img_format = _bytes_feature(box_struct.file_format.encode('utf-8'))
            else:
                img_format = _bytes_feature('depth'.encode('utf-8'))

            # normalization factor and bounding boxes
            normalization_factor_x = 640.
            normalization_factor_y = 480.

            xmin_values = [box.xmin/normalization_factor_x for box in box_struct.boundingboxes]
            xmax_values = [box.xmax/normalization_factor_x for box in box_struct.boundingboxes]
            ymin_values = [box.ymin/normalization_factor_y for box in box_struct.boundingboxes]
            ymax_values = [box.ymax/normalization_factor_y for box in box_struct.boundingboxes]

            label_values = [box.label for box in box_struct.boundingboxes]
            label_text = ['person' for box in box_struct.boundingboxes]
            # make sure that boxes and labels have the same length
            if not (len(xmin_values) == len(xmax_values) == len(ymin_values) == len(ymax_values) == len(label_values)):
                raise ValueError('All boxes must have the same size')
            if len(box_struct.boundingboxes) == 0:
                continue
            if len(xmax_values) == 0:
                raise ValueError('Number of bounding boxes should not be empty')

            feat = {
                'image/height': _int64_feature(box_struct.height),
                'image/width': _int64_feature(box_struct.width),
                'image/object/bbox/xmin': _float_feature(xmin_values),
                'image/object/bbox/xmax': _float_feature(xmax_values),
                'image/object/bbox/ymin': _float_feature(ymin_values),
                'image/object/bbox/ymax': _float_feature(ymax_values),
                'image/object/class/label': _int64_feature(label_values),
                'image/object/class/text': _bytes_feature(label_text),
                'image/encoded': _bytes_feature(image_raw),
                'image/format': img_format,
                'image/depth': _int64_feature([3]),
                'boxes/length': _int64_feature([len(box_struct.boundingboxes)]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())
            index_written_file += 1
        writer.close()
    print('Written examples: {0}'.format(index_written_file))
