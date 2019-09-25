from tensorflow.python.keras.layers import Conv2D, SeparableConv2D
from ..nn.squeezenet import squeezenet1_1

from .ssd import SSD
from .predictor import Predictor
from .config import squeezenet_ssd_config as config


def create_squeezenet_ssd_lite(num_classes, is_test=False):
    base_net = squeezenet1_1(False).features  # disable dropout layer

    source_layer_indexes = [
        12
    ]
    extras = [
        [
            Conv2D(filters=256, kernel_size=1, activation='relu'),
            SeparableConv2D(filters=512, kernel_size=3, strides=2, padding='same'),
        ],
        [
            Conv2D(filters=256, kernel_size=1, activation='relu'),
            SeparableConv2D(filters=512, kernel_size=3, strides=2, padding='same'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            SeparableConv2D(filters=256, kernel_size=3, strides=2, padding='same'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            SeparableConv2D(filters=256, kernel_size=3, strides=2, padding='same'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            SeparableConv2D(filters=256, kernel_size=3, strides=2, padding='same')
        ]
    ]

    regression_headers = [
        SeparableConv2D(filters=6 * 4, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * 4, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * 4, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * 4, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * 4, kernel_size=3, padding='same'),
        Conv2D(filters=6 * 4, kernel_size=1)
    ]

    classification_headers = [
        SeparableConv2D(filters=6 * num_classes, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * num_classes, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * num_classes, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * num_classes, kernel_size=3, padding='same'),
        SeparableConv2D(filters=6 * num_classes, kernel_size=3, padding='same'),
        Conv2D(filters=6 * num_classes, kernel_size=1),
    ]

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_squeezenet_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma)
    return predictor