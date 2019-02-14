import os
import re
import json
from os.path import expanduser

import zipfile
import datetime
import tensorflow as tf
import numpy as np

from utils import mkdir_p
from inoutdoor_dataset_download import InoutdoorDatasetDownload
from inoutdoor_versions import *
from tf_features import *
from PIL import Image


class InoutdoorDatasetWriter(object):
    feature_dict = {
        'image/height': None,
        'image/width': None,
        'image/object/bbox/id': None,
        'image/object/bbox/xmin': None,
        'image/object/bbox/xmax': None,
        'image/object/bbox/ymin': None,
        'image/object/bbox/ymax': None,
        'image/object/bbox/truncated': None,
        'image/object/bbox/occluded': None,
        'image/object/class/label/name': None,
        'image/object/class/label/id': None,
        'image/object/class/label': None,
        'image/format': None,
        'image/id': None,
        'image/source_id': None,
        'image/filename': None,
        # new
        'image/object/class/text': None,
        'image/rgb/encoded': None,
        'image/depth/encoded': None,
        'image/encoded': None,
        'image/depth': None,
        'boxes/length': None,
    }

    def get_image_sets(self):
        imagesets = dict()
        for f in os.listdir(self.image_set_definition_path):
            # check if it is a file
            if not os.path.isfile(os.path.join(
                    self.image_set_definition_path, f)):
                continue
            imagesets[f] = []
            with open(os.path.join(
                    self.image_set_definition_path, f), 'r') as setfile:
                for line in setfile.readlines():
                    imagesets[f].append(
                        line if not line.endswith('\n') else line[:-1]
                    )
        return imagesets

    def __init__(self):
        self.input_path = os.path.join(expanduser('~'), 'dataset', 'inoutdoorpeoplergbd')
        assert (os.path.exists(self.input_path))

        expected_paths = ['Images', 'Depth', 'Annotations', 'ImageSets']
        for path in expected_paths:
            if not os.path.exists(os.path.join(self.input_path, path)):
                raise ValueError('Expected subdirectory {0} does not exist. {1}'.format(
                    path, os.path.join(self.input_path, path))
                )
        self.tracking_path = os.path.join(self.input_path, 'Annotations')
        self.rgb_path = os.path.join(self.input_path, 'Images')
        self.depth_path = os.path.join(self.input_path, 'DepthJet')
        self.image_set_definition_path = os.path.join(self.input_path, 'ImageSets')
        self.dataset_path = self.input_path
        self.image_sets = self.get_image_sets()

    @staticmethod
    def feature_dict_description(type='feature_dict'):
        """
        Get the feature dict. In the default case it is filled with all the keys and the items set to None. If the
        type=reading_shape the shape description required for reading elements from a tfrecord is returned)
        :param type: (anything = returns the feature_dict with empty elements, reading_shape = element description for
        reading the tfrecord files is returned)
        :return:
        """
        obj = InoutdoorDatasetWriter.feature_dict
        if type == 'reading_shape':
            obj['image/height'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/width'] = tf.FixedLenFeature((), tf.int64, 1)
            obj['image/object/bbox/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/bbox/xmin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/xmax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymin'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/ymax'] = tf.VarLenFeature(tf.float32)
            obj['image/object/bbox/truncated'] = tf.VarLenFeature(tf.string)
            obj['image/object/bbox/occluded'] = tf.VarLenFeature(tf.string)
            obj['image/encoded'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/format'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/filename'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/source_id'] = tf.FixedLenFeature((), tf.string, default_value='')
            obj['image/object/class/label/id'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label'] = tf.VarLenFeature(tf.int64)
            obj['image/object/class/label/name'] = tf.VarLenFeature(tf.string)
            #
            obj['image/object/class/label'] = tf.FixedLenFeature((), tf.int64, 1),
            obj['image/object/class/text'] = tf.FixedLenFeature((), tf.string, default_value=''),
            obj['image/rgb/encoded'] = tf.FixedLenFeature((), tf.string, default_value=''),
            obj['image/depth/encoded'] = tf.FixedLenFeature((), tf.string, default_value=''),
            obj['image/encoded'] = tf.FixedLenFeature((), tf.string, default_value=''),
            obj['image/depth'] =  tf.FixedLenFeature((), tf.int64, 1)
            obj['boxes/length'] = tf.FixedLenFeature((), tf.int64, 1)
        return obj


    def unzip_file_to_folder(self, filename, folder, remove_file_after_creating=True):
        assert(os.path.exists(filename) and os.path.isfile(filename))
        assert(os.path.exists(folder) and os.path.isdir(folder))
        with zipfile.ZipFile(filename, 'r') as zf:
            zf.extractall(folder)
        if remove_file_after_creating:
            print('\nRemoving file: {0}'.format(filename))
            os.remove(folder)

    def get_image_label_folder(self, fold_type=None, version=None):
        """
        Returns the folder containing all images and the folder containing all label information
        :param fold_type:
        :param version:
        :return: Raises BaseExceptions if expectations are not fulfilled
        """

        download_folder = os.path.join(self.input_path, 'download')
        expansion_images_folder = os.path.join(self.input_path, 'Images')
        expansion_depthjet_folder = os.path.join(self.input_path, 'DepthJet')
        expansion_labels_folder = os.path.join(self.input_path, 'Annotations')
        #
        if not os.path.exists(expansion_images_folder):
            mkdir_p(expansion_images_folder)
        if not os.path.exists(expansion_depthjet_folder):
            mkdir_p(expansion_depthjet_folder)
        if not os.path.exists(expansion_labels_folder):
            mkdir_p(expansion_labels_folder)

        full_images_path = expansion_images_folder
        full_depthjet_path = expansion_depthjet_folder
        full_labels_path = expansion_labels_folder

        extract_files = True
        if len(InoutdoorDatasetDownload.filter_files(full_labels_path)) == \
                len(InoutdoorDatasetDownload.filter_files(full_images_path)):
            print('Do not check the download folder. Pictures seem to exist.')
            extract_files = False
        elif os.path.exists(download_folder):
            raise BaseException('not yet implemented')
            # files_in_directory = InoutdoorDatasetDownload.filter_files(
            #     download_folder, False, re.compile('\.zip$'))
            # if len(files_in_directory) < 2:
            #     raise BaseException('Not enough files found in {0}. All files present: {1}'.format(
            #         download_folder, files_in_directory
            #     ))
        else:
            mkdir_p(download_folder)
            raise BaseException('Download folder: {0} did not exist. It had been created. '
                                'Please put images, labels there.'.format(download_folder))

        # unzip the elements
        if extract_files:
            print('Starting to unzip the files')
            raise BaseException('Starting to unzip the files')

        if fold_type == 'test':
            return full_images_path, full_depthjet_path, None
        return full_images_path, full_depthjet_path, full_labels_path


    def _get_boundingboxes(self, annotations_for_picture_id):
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded = \
            [], [], [], [], [], [], [], [], []
        if annotations_for_picture_id is None:
            return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded
        for i, object in enumerate(annotations_for_picture_id.get('object', [])):
            if 'bndbox' not in object:
                continue
            boxid.append(i)
            xmin.append(float(object['bndbox']['xmin']))
            xmax.append(float(object['bndbox']['xmax']))
            ymin.append(float(object['bndbox']['ymin']))
            ymax.append(float(object['bndbox']['ymax']))
            label.append(object['name'])
            label_id.append(INOUTDOOR_LABELS.index(object['name']) + 1)

            truncated.append(False)
            occluded.append(False)
        return boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded


    def _get_tf_feature_dict(self, image_id, image_path, image_format, annotations):
        assert(isinstance(image_path, dict))
        boxid, xmin, xmax, ymin, ymax, label_id, label, truncated, occluded = \
            self._get_boundingboxes(annotations)
        truncated = np.asarray(truncated)
        occluded = np.asarray(occluded)

        # convert things to bytes
        label_bytes = [tf.compat.as_bytes(l) for l in label]

        default_image_path = image_path['rgb'] \
            if image_path.get('rgb', None) is not None \
            else image_path['depth']

        im = Image.open(default_image_path)
        image_width, image_height = im.size
        image_filename = os.path.basename(default_image_path)

        xmin = [x / float(image_width) for x in xmin]
        xmax = [x / float(image_width) for x in xmax]
        ymin = [y / float(image_height) for y in ymin]
        ymax = [y / float(image_height) for y in ymax]

        image_fileid = re.search('^(.*)(\.png)$', image_filename).group(1)
        assert(image_fileid == image_id)

        tmp_feat_dict = InoutdoorDatasetWriter.feature_dict
        tmp_feat_dict['image/id'] = bytes_feature(image_fileid)
        tmp_feat_dict['image/source_id'] = bytes_feature(image_fileid)
        tmp_feat_dict['image/height'] = int64_feature(image_height)
        tmp_feat_dict['image/width'] = int64_feature(image_width)
        tmp_feat_dict['image/depth'] = int64_feature([3])

        for key, item in image_path.items():
            if item is None:
                continue
            with open(item, 'rb') as f:
                tmp_feat_dict['image/{0}/encoded'.format(key)] = bytes_feature(f.read())

        tmp_feat_dict['image/format'] = bytes_feature(image_format)
        tmp_feat_dict['image/filename'] = bytes_feature(image_filename)
        tmp_feat_dict['image/object/bbox/id'] = int64_feature(boxid)
        tmp_feat_dict['image/object/bbox/xmin'] = float_feature(xmin)
        tmp_feat_dict['image/object/bbox/xmax'] = float_feature(xmax)
        tmp_feat_dict['image/object/bbox/ymin'] = float_feature(ymin)
        tmp_feat_dict['image/object/bbox/ymax'] = float_feature(ymax)
        tmp_feat_dict['image/object/bbox/truncated'] = bytes_feature(
            truncated.tobytes())
        tmp_feat_dict['image/object/bbox/occluded'] = bytes_feature(
            occluded.tobytes())
        tmp_feat_dict['image/object/class/label/id'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label'] = int64_feature(label_id)
        tmp_feat_dict['image/object/class/label/name'] = bytes_feature(
            label_bytes)

        items_to_remove = [
            key for key, item in tmp_feat_dict.items() if item is None
        ]
        for it in items_to_remove:
            del tmp_feat_dict[it]

        return tmp_feat_dict

    def _get_tf_feature(self, image_id, image_path, image_format, annotations):
        feature_dict = self._get_tf_feature_dict(
            image_id, image_path, image_format, annotations)
        return tf.train.Features(feature=feature_dict)

    def write_tfrecord(self, fold_type=None, version=None,
                       max_elements_per_file=1000, maximum_files_to_write=None,
                       write_masks=False):
        assert(version is None or version in ['rgb', 'depth', 'both'])
        assert(fold_type in self.image_sets.keys())
        assert(fold_type is not None and
               re.match('^(seq\d)\.txt$', fold_type))
        if version is None:
            version = 'rgb'
        sequence_type = re.match('^(seq\d)\.txt$', fold_type).group(1)
        output_path = os.path.join(self.input_path, 'tfrecord')

        if not os.path.exists(output_path):
            mkdir_p(output_path)

        full_images_path, full_depthjet_path, full_labels_path = \
            self.get_image_label_folder(fold_type, version)

        def get_annotation(picture_id):
            if full_labels_path is None:
                return None
            with open(os.path.join(
                    full_labels_path, picture_id + '.yml'), 'r') as f:
                import yaml
                obj = yaml.load(f.read())
                obj_annotation = obj['annotation']
                return obj_annotation

        image_filename_regex = re.compile('^(.*)\.(png)$')
        tfrecord_file_id, writer = 0, None
        tfrecord_filename_template = os.path.join(
            output_path,
            'output_modality_{modality}_'
            'sequence_{version}_'
            'split_{{iteration:06d}}.tfrecord'.format(
                modality=version,
                version=sequence_type
            ))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            files_written = 0
            for _, f in enumerate(self.image_sets[fold_type]):
                f = '{0}.png'.format(f)
                if files_written % max_elements_per_file == 0:
                    if writer is not None:
                        writer.close()
                        tfrecord_file_id += 1
                    tmp_filename_tfrecord = tfrecord_filename_template.format(
                        iteration=tfrecord_file_id)
                    print('{0}: Create TFRecord filename: {1} after '
                          'processing {2}/{3} files'.format(
                        str(datetime.datetime.now()), tmp_filename_tfrecord,
                        files_written, len(self.image_sets[fold_type])
                    ))
                    writer = tf.python_io.TFRecordWriter(
                        tmp_filename_tfrecord
                    )
                elif files_written % 250 == 0:
                    print('\t{0}: Processed file: {1}/{2}'.format(
                        str(datetime.datetime.now()),
                        files_written, len(self.image_sets[fold_type])))
                # match the filename with the regex
                m = image_filename_regex.search(f)
                if m is None:
                    print('Filename did not match regex: {0}'.format(f))
                    continue

                picture_id = m.group(1)
                picture_id_annotations = get_annotation(picture_id)

                filenames = {'rgb': None, 'depth': None}
                if version == 'rgb' or version is None:
                    filenames['rgb'] = os.path.join(full_images_path, f)
                elif version == 'depth':
                    filenames['depth'] = os.path.join(full_depthjet_path, f)
                else:
                    filenames = {
                        'rgb': os.path.join(full_images_path, f),
                        'depth': os.path.join(full_depthjet_path, f)
                    }

                feature = self._get_tf_feature(
                    picture_id, filenames, m.group(2), picture_id_annotations)
                example = tf.train.Example(features=feature)
                writer.write(example.SerializeToString())

                if maximum_files_to_write is not None:
                    if files_written < maximum_files_to_write:
                        break
                files_written += 1

            # Close the last files
            if writer is not None:
                writer.close()
