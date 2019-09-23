from collections import namedtuple
from typing import List

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Concatenate, Add, Reshape, Softmax
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from ..utils import box_utils

GraphPath = namedtuple("GraphPath", ['s0', 's1'])  #


class SSD:
    def __init__(self, num_classes: int, base_net: Model, source_layer_indexes: List[int],
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
            one_big_prediction = Concatenate()
            output = one_big_prediction([confidences, locations])
            self.ssd = Model(inputs=self.base_net.input, outputs=output)
        else:
            self.ssd = Model(inputs=self.base_net.input, outputs=[confidences, locations])

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
                if type(layer) == Add:
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

        confidences = Concatenate(1)(confidences)
        locations = Concatenate(1)(locations)

        if self.is_test:
            confidences = Softmax(axis=2)(confidences)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = Reshape((-1, self.num_classes))(confidence)

        location = self.regression_headers[i](x)
        location = Reshape((-1, 4))(location)

        return confidence, location

    def init_from_pretrained_ssd(self, weights_path):
        self.ssd.load_weights(weights_path, by_name=True)
        for layer in self.ssd.layers:
            if layer in self.classification_headers or layer in self.regression_headers:
                w = layer.get_weights()
                if len(w) > 1:  # Layer has bias
                    new_weights = []
                    new_weights.append(glorot_uniform()(w[0].shape))
                    new_weights.append(Zeros()(w[1].shape))
                else:
                    new_weights = glorot_uniform()(w.shape)
                layer.set_weights(new_weights)


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
