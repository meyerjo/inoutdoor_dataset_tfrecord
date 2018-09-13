import math
import re
from os import listdir
from os.path import isfile, join, exists

import numpy as np
from PIL import Image

from data_io.data_structs import BoundingBox, BoundingBoxStruct
from data_io.filereader.read_pgm_to_1d_depth import read_pgm_to_1d_depth
from data_io.tfrecord_encoding_png import create_boundingbox_tfrecord
from utils.colorize_depth_map import colorize_1d_depth_map


class InOutDoor_Dataset(object):
    def __init__(self, dataset_path, number_of_splits=1):
        if not exists(dataset_path):
            raise ValueError('Dataset Path does not exist: {0}'.format(
                dataset_path
            ))
        # check if all files exist
        expected_paths = ['Images', 'Depth', 'Annotations', 'ImageSets']
        for path in expected_paths:
            if not exists(join(dataset_path, path)):
                raise ValueError('Expected subdirectory {0} does not exist. {1}'.format(
                    path, join(dataset_path, path))
                )
        self.tracking_path = join(dataset_path, 'Annotations')
        self.rgb_path = join(dataset_path, 'Images')
        self.depth_path = join(dataset_path, 'DepthJet')
        self.image_set_definition_path = join(dataset_path, 'ImageSets')
        self.dataset_path = dataset_path

        self.image_sets = self.get_image_sets()
        # fix an error in the dataset which hinders the yaml load process in using python-yaml
        self.fix_yaml_version_error()

        self.image_files = [f for f in listdir(self.rgb_path) if isfile(join(self.rgb_path, f))]
        self.depth_files = [f for f in listdir(self.depth_path) if isfile(join(self.depth_path, f))]
        self.number_of_train_splits = min(number_of_splits, len(self.image_sets.keys()))
        self.annotations = self.get_all_bounding_boxes_per_filename(self.tracking_path)

    def fix_yaml_version_error(self):
        import re
        all_files_in_dir = [f for f in listdir(self.tracking_path) if isfile(join(self.tracking_path, f))]

        for file in all_files_in_dir:
            with open(join(self.tracking_path, file), 'r+') as f:
                content = f.read()
                old_content = content
                # check if YAML:1.0 is still included
                if content.startswith('%YAML:1.0'):
                    content = content.replace('%YAML:1.0', '%YAML 1.0')

                # check if the "---" after the annotation is available
                if re.match('^%YAML[\:\s]1\.0\n---\n', content) is None:
                    split_lines = content.split('\n')
                    split_lines.insert(1, '---')
                    content = '\n'.join(split_lines)

                if old_content != content:
                    f.seek(0)
                    f.write(content)
                    f.truncate()


    def get_image_sets(self):
        imagesets = dict()
        for f in listdir(self.image_set_definition_path):
            # check if it is a file
            if not isfile(join(self.image_set_definition_path, f)):
                continue
            imagesets[f] = []
            with open(join(self.image_set_definition_path, f), 'r') as setfile:
                for line in setfile.readlines():
                    imagesets[f].append(
                        line if not line.endswith('\n') else line[:-1]
                    )
        return imagesets


    def get_all_bounding_boxes_per_filename(self, folder):
        # somehow organize all the track-files
        all_files_in_dir = [f for f in listdir(folder) if isfile(join(folder, f))]
        import yaml
        imagename_to_annotation = dict()
        for f in all_files_in_dir:
            with open(join(folder, f), 'r') as r:
                try:
                    obj = yaml.load(r)
                    obj_annotation = obj['annotation']
                    if 'object' in obj_annotation:
                        boundingboxes = obj_annotation['object']
                        imagename_to_annotation[f] = boundingboxes
                    else:
                        imagename_to_annotation[f] = []
                except yaml.YAMLError as exc:
                    print(exc)
                    exit(-1)
        return imagename_to_annotation

    @staticmethod
    def open_image(path, mode):
        """
        Opens a image
        :param path:
        :param mode:
        :return: Tuple of Image-object and numpy array
        """
        if mode == 'rgb':
            orig_img = Image.open(path)
            img = np.array(orig_img)
        elif mode == 'depth':
            # load image as array, colorize array, load array as image
            M = read_pgm_to_1d_depth(path, display=False)
            img = colorize_1d_depth_map(M)
            orig_img = Image.fromarray(img)
        else:
            raise ValueError('Unknown mode: {0}'.format(mode))
        return orig_img, img


    @staticmethod
    def handle_bounding_boxes(boxes_struct, labeled_bounding_boxes, input_type='rgb'):
        for j, box in enumerate(labeled_bounding_boxes):
            class_label = int(box['name'] == 'person')  # label is saved in the last element of the row
            xmin = int(box['bndbox']['xmin']) / 1920.
            xmax = int(box['bndbox']['xmax']) / 1920.
            ymin = int(box['bndbox']['ymin']) / 1080.
            ymax = int(box['bndbox']['ymax']) / 1080.
            boxes_struct.append(BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, label=class_label, normalized=False))
        return boxes_struct

    def filter_images_by_image_id(self, image_files, ids):
        _files = []
        for file in image_files:
            if file[:-4] in ids:
                _files.append(file)
        return _files

    @staticmethod
    def _write_tfrecord(image_filenames, dataset_path, mode, output_filename, annotations):
        print('Writing to: {0}\nLogging every 100 images'.format(output_filename))
        bb_box = []

        dataset_path_modality_one = dataset_path[0]
        dataset_path_modality_two = dataset_path[1] if len(dataset_path) > 1 else None

        for i, file in enumerate(image_filenames):
            if i % 100 == 0:
                print('{0}/{1} images'.format(i, len(image_filenames)))
            file_path_modality_one = join(dataset_path_modality_one, file)
            file_path_modality_two = join(dataset_path_modality_two, file) if dataset_path_modality_two is not None else None

            # handle the bounding boxes
            file_name_without_extension = file[:-4]
            bounding_box_descriptions = []
            if (file_name_without_extension + '.yml') in annotations:
                bounding_box_descriptions = annotations[file_name_without_extension + '.yml']

            # iterate across all bounding boxes
            boxes = InOutDoor_Dataset.handle_bounding_boxes([], bounding_box_descriptions, input_type=mode)

            bb = BoundingBoxStruct(img_path_rgb=file_path_modality_one,
                                   img_path_depth=file_path_modality_two,
                                   boundingboxes=boxes, height=1080, width=1920)
            bb_box.append(bb)

        load_function = lambda x: InOutDoor_Dataset.open_image(x, mode)
        create_boundingbox_tfrecord(bb_box, output_filename, load_function)

    def parser(self, mode, output_stem, output_mode='dual'):
        print('Only dual output mode implemented at the moment')
        if mode == 'rgb':
            image_filenames = self.image_files
            general_path = self.rgb_path
        elif mode == 'depth':
            image_filenames = self.depth_files
            general_path = self.depth_path
        else:
            raise ValueError('Unknown mode: {0}'.format(mode))

        if output_mode == 'dual':
            general_path = [self.rgb_path, self.depth_path]

            if len(list(set(self.image_files).union(set(self.depth_files)))) != len(self.image_files):
                print('Union of lists should not get larger')
                exit(-1)
            image_filenames = self.image_files
        else:
            general_path = [general_path]

        IMAGES_PER_FILE = 500
        for item, files in self.image_sets.items():
            m = re.search('^seq(\d)\.txt$', item)
            r = m.group(1)

            print('Creating training/test split: {0}'.format(r))

            sequence_filenames = self.filter_images_by_image_id(image_filenames, files)

            output_filename = output_stem + '_seq{sequence_id}_split{{split_id}}.tfrecord'.format(
                sequence_id=r)

            number_of_splits = int(math.ceil(len(sequence_filenames)/float(IMAGES_PER_FILE)))

            for split_i in range(0, number_of_splits):
                tmp_filename = output_filename.format(split_id=split_i)
                tmp_filenames = sequence_filenames[split_i*IMAGES_PER_FILE:(split_i+1)*IMAGES_PER_FILE]

                InOutDoor_Dataset._write_tfrecord(
                    image_filenames=tmp_filenames, dataset_path=general_path, mode=mode,
                    output_filename=tmp_filename, annotations=self.annotations
                )
