import numpy as np
import tensorflow as tf

from .data_preprocessing import PredictionTransform
from ..utils import box_utils
from ..utils.misc import Timer
from vision.utils.box_utils_numpy import convert_locations_to_boxes, center_form_to_corner_form


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, config=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.config = config
        self.priors = config.priors

        self.sigma = sigma

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        height, width, _ = image.shape
        image = self.transform(image)
        image = np.expand_dims(image, 0)

        inputs = {'input': image}

        self.timer.start()
        self.net.run(inputs)
        print("Inference time: ", self.timer.end())

        scores = self.net.tensor('output2').map()
        locations = self.net.tensor('output1').map()

        boxes = convert_locations_to_boxes(locations, self.priors, self.config.center_variance,
                                           self.config.size_variance)
        boxes = center_form_to_corner_form(boxes)
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, scores.shape[1]):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = tf.concat([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])

        if not picked_box_probs:
            return tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32), tf.constant([],
                                                                                                     dtype=tf.float32)
        picked_box_probs_temp = tf.concat(picked_box_probs, axis=0)
        picked_box_probs = [picked_box_probs_temp[:, 0] * width, picked_box_probs_temp[:, 1] * height,
                            picked_box_probs_temp[:, 2] * width, picked_box_probs_temp[:, 3] * height,
                            picked_box_probs_temp[:, 4]]
        picked_box_probs = tf.stack(picked_box_probs, axis=1)

        return picked_box_probs[:, :4], tf.constant(picked_labels), picked_box_probs[:, 4]
