import logging
import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python import FixedLenFeature, VarLenFeature, parse_single_example


class RecordDataset(Sequence):

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False,
                 batch_size=32, buffer_size=None, shuffle=True):
        """
        Dataset for TFRecord data.
        Args:
            root: the root of the TFRecord, the directory contains the following files:
                label_map.txt, train.record, val.record, num_train.txt, num_val.txt
        """
        self.root = pathlib.Path(root)
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "val.record"
            if os.path.isfile(self.root / "num_val.txt"):
                with open(self.root / "num_val.txt", 'r') as f:
                    self.num_records = int(f.read())
        else:
            image_sets_file = self.root / "train.record"
            if os.path.isfile(self.root / "num_val.txt"):
                with open(self.root / "num_train.txt", 'r') as f:
                    self.num_records = int(f.read())

        if buffer_size is None:
            buffer_size = self.num_records

        self.keys_to_features = {
            'image/encoded':
                FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename':
                FixedLenFeature((), tf.string, default_value=''),
            'image/source_id':
                FixedLenFeature((), tf.string, default_value=''),
            'image/height':
                FixedLenFeature((), tf.int64, default_value=1),
            'image/width':
                FixedLenFeature((), tf.int64, default_value=1),
            # Object boxes and classes.
            'image/object/bbox/xmin':
                VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                VarLenFeature(tf.float32),
            'image/object/class/label':
                VarLenFeature(tf.int64),
            'image/object/class/text':
                VarLenFeature(tf.string),
            'image/object/difficult':
                VarLenFeature(tf.int64),
        }

        self.dataset = TFRecordDataset([str(image_sets_file)])
        if shuffle:
            self.dataset.shuffle(buffer_size=buffer_size)
        self.dataset = self.dataset.map(self.parse_sample)
        self.keep_difficult = keep_difficult
        self.num_batches = self.num_records // self.batch_size

        # if the labels file exists, read in the class names
        label_file_name = self.root / "label_map.txt"

        self._get_classes(label_file_name)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _get_classes(self, label_file_name):
        if os.path.isfile(label_file_name):
            classes = []
            with open(label_file_name, 'r') as infile:
                first_line = infile.readline()
                if first_line.rstrip() == 'item {':
                    parse_tfrecord = True
                else:
                    parse_tfrecord = False
                if not parse_tfrecord:  # Classes are a text file with class label on each line
                    classes.append(first_line)
                    for line in infile:
                        classes.append(line.rstrip())
                else:  # Classes are in tfrecord format
                    for line in infile:
                        line = line.strip()
                        if line.startswith('name: '):
                            line = line.replace('\'', '')
                            classes.append(line[5:].strip())

            if 'BACKGROUND' not in classes:
                classes.insert(0, 'BACKGROUND')

            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))
        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
                                'aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor')

    def __getitem__(self, idx):
        inputs, target1, target2 = [], [], []
        for sample in self.dataset.take(self.batch_size):
            boxes, labels, is_difficult = self._get_annotation(sample)
            if not self.keep_difficult and is_difficult:
                boxes = boxes[is_difficult == 0]
                labels = labels[is_difficult == 0]
            image = self._read_image(sample)
            if self.transform:
                image, boxes, labels = self.transform(image, boxes, labels)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
            inputs.append(image)
            target1.append(boxes.numpy())
            target2.append(labels.numpy())

        tmp_inputs = np.array(inputs, dtype=np.float32)
        tmp_target1 = np.array(target1)
        tmp_target2 = np.array(target2)
        tmp_target2 = np.expand_dims(tmp_target2, 2)
        tmp_target = np.concatenate([tmp_target1, tmp_target2], axis=2)
        return tmp_inputs, tmp_target

    def __len__(self):
        return int(np.ceil(self.num_records / float(self.batch_size)))

    def _get_annotation(self, sample):
        # Get info about each object in image
        num_objects = sample['image/object/bbox/xmax'].shape[0]
        boxes = []
        labels = []
        is_difficult = []
        height = sample['image/height'].numpy()
        width = sample['image/width'].numpy()
        for i in range(num_objects):
            # Undo bbox coord normalization
            x_max = int(sample['image/object/bbox/xmax'].values[i].numpy() * width)
            x_min = int(sample['image/object/bbox/xmin'].values[i].numpy() * width)
            y_max = int(sample['image/object/bbox/ymax'].values[i].numpy() * height)
            y_min = int(sample['image/object/bbox/ymin'].values[i].numpy() * height)
            boxes.append([x_min, y_min, x_max, y_max])

            labels.append(sample['image/object/class/label'].values[i].numpy())
            # is_difficult.append(sample['image/object/difficult'].values[i].numpy())

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, sample):
        return tf.image.decode_image(sample['image/encoded']).numpy()

    def parse_sample(self, data_record):
        sample = parse_single_example(data_record, self.keys_to_features)
        return sample
