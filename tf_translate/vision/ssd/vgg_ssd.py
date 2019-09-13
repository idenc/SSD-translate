from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Conv2D, ReLU, BatchNormalizationV2

from .config import vgg_ssd_config as config
from .predictor import Predictor
from .ssd import SSD


# from ..nn.vgg import vgg


def create_vgg_ssd(num_classes, is_test=False, is_train=False):
    base_net = VGG16(input_shape=(config.image_size, config.image_size, 3),
                     include_top=False)

    source_layer_indexes = [
        (23, BatchNormalizationV2()),
        len(base_net),
    ]
    extras = [
        [
            Conv2D(filters=256, kernel_size=1),
            ReLU(),
            Conv2D(filters=512, kernel_size=3, strides=2, padding="same"),
            ReLU()
        ],
        [
            Conv2D(filters=128, kernel_size=1),
            ReLU(),
            Conv2D(filters=256, kernel_size=3, strides=2, padding="same"),
            ReLU()
        ],
        [
            Conv2D(filters=128, kernel_size=1),
            ReLU(),
            Conv2D(filters=256, kernel_size=3),
            ReLU()
        ],
        [
            Conv2D(filters=128, kernel_size=1),
            ReLU(),
            Conv2D(filters=256, kernel_size=3),
            ReLU()
        ]
    ]

    regression_headers = [
        Conv2D(filters=4 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=6 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=4 * 4, kernel_size=3, padding="same"),
        Conv2D(filters=4 * 4, kernel_size=3, padding="same"),
        # TODO: change to kernel_size=1, padding=0?
    ]

    classification_headers = [
        Conv2D(filters=4 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=6 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=4 * num_classes, kernel_size=3, padding="same"),
        Conv2D(filters=4 * num_classes, kernel_size=3, padding="same"),
        # TODO: change to kernel_size=1, padding=0?
    ]

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, is_train=is_train)


def create_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
