import argparse
import os

from dataset.inoutdoor_dataset import InOutDoor_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/meyerjo/dataset/inoutdoorpeoplergbd/',
                    help='Directory where to find ')

parser.add_argument('--mode', type=str, help='The mode which should be converted to a tfrecord file')

parser.add_argument('--tfrecord_output', type=str,
                    default='/home/meyerjo/dataset/inoutdoorpeoplergbd/OutputTfRecord/rgbd_inoutdoor_',
                    help='TFRecord Output File')

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    path_raw_label = FLAGS.dataset_dir

    output_stem = FLAGS.tfrecord_output + FLAGS.mode
    print('Converting InOutDoor Data to tfrecord')
    print('Path: {0}'.format(path_raw_label))
    print('Mode: {0}'.format(FLAGS.mode))
    print('Output-Stem: {0}'.format(output_stem))

    dataset = InOutDoor_Dataset(path_raw_label, number_of_splits=4)
    dataset.parser(FLAGS.mode, output_stem)
