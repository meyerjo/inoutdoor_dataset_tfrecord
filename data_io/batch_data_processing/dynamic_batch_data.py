import tensorflow as tf

from data_io.tfrecord_decoding import repeat, repeat_v2


def process_dynamic_batches(image, boundingboxes, labels, number_of_boxes, *args, **kwargs):
    """
    Shall handle the image, boundingboxes, labels, number_of_boxes for dynamic sized batches
    :param image:
    :param boundingboxes:
    :param labels:
    :param number_of_boxes:
    :return:
    """
    DEBUG = kwargs.get('DEBUG', False)
    normalize = kwargs.get('normalize', True)

    if DEBUG:
        image = tf.Print(image, [], 'Shape of "Number of boxes": ' + str((number_of_boxes.get_shape())))
    g = tf.get_default_graph()

    if DEBUG:
        number_of_boxes = tf.Print(number_of_boxes, [], '-----------------------------------------------')
        labels = tf.Print(labels, [], '-----------------------------------------------')
    # get the static numbers
    number_of_boxes = tf.squeeze(number_of_boxes)
    # argmax returns the index with the maximum value

    if DEBUG:
        number_of_boxes = tf.Print(number_of_boxes, [number_of_boxes, tf.size(number_of_boxes)], 'Number of boxes, and Size: ', summarize=100)
    max_number_of_boxes_index = tf.argmax(number_of_boxes, dimension=0)
    max_number_of_boxes = number_of_boxes[max_number_of_boxes_index]

    if DEBUG:
        max_number_of_boxes = tf.Print(max_number_of_boxes, [max_number_of_boxes], 'Maximum: ', summarize=100)
    # sum the number of boxes
    total_number_of_boxes = tf.reduce_sum(number_of_boxes)
    total_number_of_images = tf.size(number_of_boxes)
    # total boxes: number of images * the maximum number of boxes
    # (if less then number-of-boxes are in a batch these are filled with -1)
    total_boxes = total_number_of_images * max_number_of_boxes

    if DEBUG:
        total_number_of_images = tf.Print(total_number_of_images, [total_number_of_images], 'Total images: ')
        total_boxes = tf.Print(total_boxes, [total_boxes], 'Total boxes: ')


    if DEBUG:
        labels = tf.Print(labels, [labels], 'Labels (before squeeze): ', summarize=100)
    labels = tf.squeeze(labels, axis=1)

    if DEBUG:
        labels = tf.Print(labels, [labels], 'Labels (after squeeze): ', summarize=100)

    boundingboxes = tf.squeeze(boundingboxes, axis=1)
    total_boxes = tf.Print(total_boxes, [total_boxes], 'Total boxes: ')
    boundingboxes = tf.reshape(boundingboxes, shape=(total_boxes, 4))

    #import ipdb; ipdb.set_trace()
    with g.control_dependencies([labels, total_boxes, number_of_boxes, max_number_of_boxes]):
        labels = tf.reshape(labels, shape=(-1, 1))

        if DEBUG:
            total_boxes = tf.Print(total_boxes, [total_boxes], 'Total boxes1: ', summarize=100)
        valid_labels = tf.greater_equal(labels, 0)
        labels = tf.boolean_mask(labels, valid_labels)

        if DEBUG:
            labels = tf.Print(labels, [labels], 'Labels (masked): ', summarize=100)
            labels = tf.Print(labels, [labels], 'Labels (reshape): ', summarize=100)

        valid_labels_boxes = tf.squeeze(tf.stack([valid_labels, valid_labels, valid_labels, valid_labels], axis=1), 2)
        boundingboxes = tf.boolean_mask(boundingboxes, valid_labels_boxes)
        boundingboxes = tf.reshape(boundingboxes, shape=(total_number_of_boxes, 4))

        image_ids = tf.range(0, total_number_of_images)
        box_ind = repeat_v2(image_ids, number_of_boxes, total_boxes)

        if DEBUG:
            box_ind = tf.Print(box_ind, [box_ind], 'Box Ind:', summarize=100)

        reshape_resize = [64, 128]
        if normalize:
            # normalize the coordinates of the bounding box
            image_size_tensor = tf.tile([480., 640.], [2])
            boundingboxes = tf.div(boundingboxes, image_size_tensor)


        images = tf.image.crop_and_resize(image, boundingboxes, box_ind, reshape_resize)
        tf.summary.image('cropped_images', images)
        if DEBUG:
            images = tf.Print(images, [images], 'Images: ' + str(images.get_shape()))
            labels = tf.Print(labels, [labels], 'Labels: ' + str(labels.get_shape()))
        return images, boundingboxes, labels, number_of_boxes
