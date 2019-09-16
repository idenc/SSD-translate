from tensorflow.python.keras.layers import Conv2D, ReLU, ZeroPadding2D

from vision.nn.mobilenet import MobileNet
from .config import mobilenetv1_ssd_config as config
from .predictor import Predictor
from .ssd import SSD


def create_mobilenetv1_ssd(num_classes, is_test=False, is_train=False):
    base_net = MobileNet(input_shape=(config.image_size, config.image_size, 3),
                         include_top=False)  # disable dropout layer

    source_layer_indexes = [
        73,
        85,
    ]

    extras = [
        [
            Conv2D(filters=256, kernel_size=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=512, kernel_size=3, strides=2, padding="valid", activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, strides=2, padding="valid", activation='relu'),
        ]
    ]

    regression_headers = [
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),  # TODO: change to kernel_size=1, padding=0?
    ]

    classification_headers = [
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),  # TODO: change to kernel_size=1, padding=0?
    ]

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, is_train=is_train)


def create_mobilenetv1_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma)
    return predictor
