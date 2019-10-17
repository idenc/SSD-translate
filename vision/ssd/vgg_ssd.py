from tensorflow.python.keras.layers import Conv2D, BatchNormalizationV2, MaxPool2D, ZeroPadding2D
from tensorflow.python.keras.models import Model

from vision.nn.vgg import VGG16
from .config import vgg_ssd_config as config
from .predictor import Predictor
from .ssd import SSD


def create_vgg_ssd(num_classes, is_test=False, is_train=False):
    base_net = VGG16(input_shape=(config.image_size, config.image_size, 3),
                     include_top=False, weights=None)
    # Add extra SSD layers
    vgg_output = base_net.output
    x = MaxPool2D(pool_size=3, strides=1, padding="same")(vgg_output)
    x = Conv2D(filters=1024, kernel_size=3, padding="same", dilation_rate=(6, 6), activation='relu')(x)
    output = Conv2D(filters=1024, kernel_size=1, padding="same", activation='relu')(x)
    base_net = Model(inputs=base_net.inputs, outputs=output)

    source_layer_indexes = [
        (14, BatchNormalizationV2(epsilon=1e-5)),
        len(base_net.layers),
    ]
    extras = [
        [
            Conv2D(filters=256, kernel_size=1, padding='same', activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides=2, padding="same", activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, padding="same", activation='relu'),
            ZeroPadding2D(padding=1),
            Conv2D(filters=256, kernel_size=3, strides=2, activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
            Conv2D(filters=256, kernel_size=3, activation='relu'),
        ],
        [
            Conv2D(filters=128, kernel_size=1, padding='same', activation='relu'),
            Conv2D(filters=256, kernel_size=3, activation='relu'),
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


def create_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma)
    return predictor
