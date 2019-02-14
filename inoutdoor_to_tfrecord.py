import argparse
import os

from inoutdoor_dataset.inoutdoor_dataset_writer import InoutdoorDatasetWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str,
                    default='/home/meyerjo/dataset/inoutdoorpeoplergbd/',
                    help='Directory where to find ')

parser.add_argument('--mode', type=str, choices=['rgb', 'depth', 'both'],
                    help='The mode which should be '
                         'converted to a tfrecord file')

parser.add_argument('--elements_per_tfrecord', type=int, default=500,
                    help='Number of Pictures per tfrecord file. '
                         'Multiple files help in the shuffling process.')

parser.add_argument('--tfrecord_output', type=str,
                    default=os.path.expanduser(
                        '~/dataset/inoutdoorpeoplergbd/'
                        'OutputTfRecord/rgbd_inoutdoor_'),
                    help='TFRecord Output File')

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    path_raw_label = FLAGS.dataset_dir

    output_stem = FLAGS.tfrecord_output + FLAGS.mode
    print('Converting InOutDoor Data to tfrecord')
    print('Path: {0}'.format(path_raw_label))
    print('Mode: {0}'.format(FLAGS.mode))
    print('Output-Stem: {0}'.format(output_stem))
    writer = InoutdoorDatasetWriter()
    writer.write_tfrecord(
        fold_type='seq0.txt', version=FLAGS.mode,
        max_elements_per_file=FLAGS.elements_per_tfrecord)
    writer.write_tfrecord(
        fold_type='seq1.txt', version=FLAGS.mode,
        max_elements_per_file=FLAGS.elements_per_tfrecord)
    writer.write_tfrecord(
        fold_type='seq2.txt', version=FLAGS.mode,
        max_elements_per_file=FLAGS.elements_per_tfrecord)
    writer.write_tfrecord(
        fold_type='seq3.txt', version=FLAGS.mode,
        max_elements_per_file=FLAGS.elements_per_tfrecord)
