import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.data import TFRecordDataset

import tensorflow as tf
import cv2
import numpy as np


class RecordDataset(Sequence):

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False,
                 batch_size=32):
        """
        Dataset for TFRecord data.
        Args:
            root: the root of the TFRecord, the directory contains the following files:
                label_map.txt, train.record, val.record.
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

        self.dataset = TFRecordDataset(image_sets_file)
        self.keep_difficult = keep_difficult
        self.num_batches = len(self.ids) // self.batch_size

        # if the labels file exists, read in the class names
        label_file_name = self.root / "label_map.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
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

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.feature_list = {
            'image/height': tf.FixedLenFeature([], tf.int64)
        }
        # self.feature_list = {
        #     'image/height': tf.FixedLenFeature(),
        #     'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        #     'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        #     'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        #     'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        #     'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        #     'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        #     'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        #     'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        #     'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        #     'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        #     'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        # }

    def __getitem__(self, idx):
        inputs, target1, target2 = [], [], []
        end = (idx + 1) * self.batch_size
        if end >= len(self.ids):
            end = len(self.ids) - 1
        idxs = np.arange(idx * self.batch_size, end)
        np.random.shuffle(idxs)
        for j, i in enumerate(idxs):
            image_id = self.ids[i]
            boxes, labels, is_difficult = self._get_annotation(image_id)
            if not self.keep_difficult:
                boxes = boxes[is_difficult == 0]
                labels = labels[is_difficult == 0]
            image = self._read_image(image_id)
            if self.transform:
                image, boxes, labels = self.transform(image, boxes, labels)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
            inputs.append(image)
            target1.append(boxes)
            target2.append(labels)

            if len(target1) == self.batch_size:
                tmp_inputs = np.array(inputs, dtype=np.float32)
                tmp_target1 = np.array(target1)
                tmp_target2 = np.array(target2)
                tmp_target2 = np.expand_dims(tmp_target2, 2)
                tmp_target = np.concatenate([tmp_target1, tmp_target2], axis=2)
                return tmp_inputs, tmp_target

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with classes in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

if __name__ == 'main':
    record = RecordDataset(r'D:\train\tf_records')