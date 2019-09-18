from collections import namedtuple
from typing import List

import tensorflow as tf

from ..utils import box_utils

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD:
    def __init__(self, num_classes: int, base_net: tf.keras.Model, source_layer_indexes: List[int],
                 extras: List, classification_headers: List,
                 regression_headers: List, is_test=False, config=None, is_train=False):
        """
        Compose a SSD model using the given components.
        """
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = [t[1] for t in source_layer_indexes
                                     if isinstance(t, tuple) and not isinstance(t, GraphPath)]
        if is_test:
            self.priors = config.priors

        # input = tf.keras.Input(shape=(300, 300, 3), name="input", dtype=tf.float32)
        confidences, locations = self.call(self.base_net.input)

        if is_train:
            one_big_prediction = tf.keras.layers.Concatenate()
            output = one_big_prediction([confidences, locations])
            self.ssd = tf.keras.models.Model(inputs=self.base_net.input, outputs=output)
        else:
            self.ssd = tf.keras.models.Model(inputs=self.base_net.input, outputs=[confidences, locations])

    def get_start_layer(self):
        if isinstance(self.source_layer_indexes[0], GraphPath):
            return self.source_layer_indexes[0].s0[0]
        elif isinstance(self.source_layer_indexes[0], tuple):
            return self.source_layer_indexes[0][0]
        else:
            return self.source_layer_indexes[0]

    def call(self, x):
        confidences = []
        locations = []
        start_layer_index = self.get_start_layer()
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index, end_layer_index_slice = \
                    end_layer_index.s0[0], end_layer_index.s0[1]
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            x = self.base_net.layers[start_layer_index - 1].output
            for layer in self.base_net.layers[start_layer_index: end_layer_index]:
                if type(layer) == tf.keras.layers.Add:
                    x = layer(layer.input)
                else:
                    x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = self.base_net.layers[end_layer_index:end_layer_index_slice]
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net.layers[end_layer_index:]:
            if type(layer) == tf.keras.layers.Add:
                # x2 = self.base_net.layers[-9]
                print()
            x = layer(x)

        for sequence in self.extras:
            if type(sequence) == list:
                for layer in sequence:
                    x = layer(x)
            else:
                x = sequence(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = tf.keras.layers.Concatenate(1)(confidences)
        locations = tf.keras.layers.Concatenate(1)(locations)

        if self.is_test:
            confidences = tf.keras.layers.Softmax(axis=2)(confidences)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = tf.keras.layers.Reshape((-1, self.num_classes))(confidence)

        location = self.regression_headers[i](x)
        location = tf.keras.layers.Reshape((-1, 4))(location)

        return confidence, location

    def load(self, model):
        self.ssd = tf.keras.models.load_model(model)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels
