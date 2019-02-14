import os
import random
import re
from os.path import expanduser

from inoutdoor_dataset_writer import InoutdoorDatasetWriter, InoutdoorDatasetDownload
from scope_wrapper import scope_wrapper
from tf_features import *
from utils import mkdir_p


@scope_wrapper
def _recover_boundingboxes(features):
    """ Creates a list of boxes [(ymin,xmin, ...), ...] from features. """
    ymin = features['image/object/bbox/ymin'].values
    xmin = features['image/object/bbox/xmin'].values
    ymax = features['image/object/bbox/ymax'].values
    xmax = features['image/object/bbox/xmax'].values
    return tf.transpose([ymin, xmin, ymax, xmax])


class InoutdoorDatasetReader():

    def get_folders(self):
        folders = InoutdoorDatasetDownload.filter_folders(self.input_path, return_relative=True)
        for key, item in self.folders_dict.items():
            tmp_regex = item['regex']
            for f in folders:
                m = tmp_regex.search(f)
                if m is None:
                    continue
                self.folders_dict[key]['folders'].append(
                    (f, os.path.join(self.input_path, f))
                )
        for key, item in self.folders_dict.items():
            if item['folders']:
                break

    def __init__(self, batch_size=1, epochs=1, threads=4, parallel_reads=2,
                 num_chained_buffers=2, buffer_size=128, modality='rgb'):
        self.folders_dict = dict()
        self.batch_size = batch_size
        self.epochs = epochs
        self.threads = threads

        self.parallel_reads = parallel_reads
        self.num_chained_buffers = num_chained_buffers
        self.buffer_size = buffer_size
        self.modality = modality

        # self.input_path = os.path.join(expanduser('~'), '.inoutdoor', 'tfrecord')
        self.input_path = os.path.join(
            expanduser('~'), 'dataset', 'inoutdoorpeoplergbd', 'tfrecord')
        self.DEFAULT_TEST_SET = 3
        if not os.path.exists(self.input_path):
            print('TFRecord path does not exists: {0}. '
                  'First create the tfrecord file.'.format(self.input_path))
            exit(-1)

    def generate_dataset(self, filenames, parsing_fn=None, shape_fn=None,
                         parallel_reads=2, num_chained_buffers=2,
                         buffer_size=128, repeat=1, num_threads=4,
                         batch_size=1):
        """
        Generator a dataset based on tfrecord files
        :param filenames:
        :param parsing_fn:
        :param shape_fn: array of shapes, which are returned from the parsing_fn
        :param parallel_reads: int - Parallel calls to the parsing function (default=2)
        :param num_chained_buffers: int - Number of chained shuffle operations
        :param buffer_size: int - Buffer size used for the shuffling of elements (default=128)
        :param repeat: int - Number of times the dataset contents are repeated (default=1)
        :param num_threads: int - Number of threads (default=4)
        :param batch_size: int - Batch Size (default=1)
        :return:
        """
        assert(filenames != [])
        assert(parsing_fn is not None and shape_fn is not None)
        random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        # TODO: do the interleaving: http://www.moderndescartes.com/essays/shuffle_viz/
        dataset = dataset.repeat(repeat)
        dataset = dataset.map(parsing_fn, num_parallel_calls=num_threads)
        for i in range(num_chained_buffers):
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.prefetch(batch_size * 30)
        dataset = dataset.padded_batch(batch_size, padded_shapes=shape_fn)
        return dataset

    @staticmethod
    def parsing_boundingboxes(serialized_example, output='tensors', modality='rgb'):
        """

        :param serialized_example:
        :param output: (anything, 'shape', 'labels')
        :return:
        """
        if output == 'shape':
            return ([None, None, 3], [None, 4], [None], [], [None], [2], )
        if output == 'labels':
            return 'image', 'bboxes', 'bbox_labels', 'image_ids', 'box_ids', 'image_shape'

        feature_def = InoutdoorDatasetWriter.feature_dict_description('reading_shape')
        if modality == 'rgb':
            del feature_def['image/depth/encoded']
        else:
            del feature_def['image/rgb/encoded']
        del feature_def['image/encoded']

        features = tf.parse_single_example(serialized_example, feature_def)

        image = tf.image.decode_png(features['image/{0}/encoded'.format(
            modality
        )], channels=3)
        boundingboxes = _recover_boundingboxes(features)

        image_shape = tf.convert_to_tensor([features['image/width'], features['image/height']])
        image_ids = features['image/id']
        # box_ids = tf.cast(features['image/object/bbox/id'].values, tf.int64)
        # boundingbox_labels = tf.cast(features['image/object/class/label'].values, tf.int64)
        box_ids = tf.cast(features['image/object/bbox/id'], tf.int64)
        boundingbox_labels = tf.cast(features['image/object/class/label'], tf.int64)
        return tf.stop_gradient(image), tf.stop_gradient(boundingboxes), \
               tf.stop_gradient(boundingbox_labels), tf.stop_gradient(image_ids), \
               tf.stop_gradient(box_ids), tf.stop_gradient(image_shape)

    def get_version_folder(self, fold_type, version):
        return self.input_path

    def load_boundingbox_data(self, fold_type, version, download=False):
        train_dir = self.get_version_folder(fold_type, version)
        test_set_file = self.DEFAULT_TEST_SET
        if version is not None and re.search('^[0123]$', str(version)):
            test_set_file = int(version)
        filename_regex = re.compile('(.*?)_seq{0}_(.*?)tfrecord$'.format(
            '[^{0}]'.format(test_set_file) if fold_type == 'train'
            else '{0}'.format(test_set_file)
        ))
        filenames = InoutdoorDatasetDownload.filter_files(train_dir, False, filename_regex)
        if len(filenames) == 0 and download:
            print('No TFRecord files found: {0}\n\t'
                  'Build tfrecords.'.format(train_dir))
            exit(-1)

        parser = lambda x: InoutdoorDatasetReader.parsing_boundingboxes(x, modality=self.modality)
        shape = InoutdoorDatasetReader.parsing_boundingboxes(None, 'shape')
        dataset = self.generate_dataset(
            filenames, parser, shape,
            self.parallel_reads, self.num_chained_buffers,
            self.buffer_size, self.epochs, self.epochs, self.batch_size)
        return dataset.make_one_shot_iterator().get_next(name='sample_tensor')

    def load_data_bbox(self, fold_type=None, version=None, download=False, write_masks=False):
        # TODO: verify this
        return self.load_boundingbox_data(fold_type, version, download)

    def load_train_data_bbox(self, version=None, download=True):
        return self.load_data_bbox(fold_type='train', version=version, download=download)

    def load_val_data_bbox(self, version=None, download=True):
        raise BaseException('No validation set available')

    def load_test_data_bbox(self, version=None, download=True):
        return self.load_data_bbox(fold_type='test', version=version, download=download)

    @staticmethod
    def find_corrupt_tfrecords(train_dir, verbose=True):
        train_files = sorted(
            [os.path.join(train_dir, f)
             for f in os.listdir(train_dir) if os.path.isfile(f)])
        total_images = []
        corrupted_images = []
        for f_i, file in enumerate(train_files):
            print(f_i)
            try:
                sum_elements = sum([1 for _ in tf.python_io.tf_record_iterator(file)])
                total_images += sum_elements
                if verbose:
                    print('{0}: {1} elements'.format(file, sum_elements))
            except BaseException as e:
                corrupted_images.append(file)
                if verbose:
                    print('{0}: Corrupted {1}'.format(file, str(e)))
        if verbose:
            for f in corrupted_images:
                print('Corrupted file: {0}'.format(f))
        return corrupted_images
