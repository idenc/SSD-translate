import argparse
import logging
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from vision.datasets.open_images import OpenImagesDataset
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.concat_datasets import ConcatDataset
from vision.datasets.tfrecord_dataset import RecordDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.config import vgg_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import str2bool, Timer, store_labels
from vision.utils.sgdr import SGDRScheduler


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.i = 0
        self.x = np.array([])
        self.losses = np.array([])
        self.val_losses = np.array([])

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(logs)
        self.x = np.append(self.x, self.i)
        self.losses = np.append(self.losses, logs.get('loss'))
        self.val_losses = np.append(self.val_losses, logs.get('val_loss'))
        val_mask = np.isfinite(self.val_losses.astype(np.double))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss", marker='o')
        if np.any(val_mask):
            plt.plot(self.x[val_mask], self.val_losses[val_mask], label="val_loss", marker='o')
        else:
            plt.plot(self.x, self.val_losses, label="val_loss", marker='o')
        plt.ion()
        plt.legend()
        plt.show()
        plt.pause(0.001)


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Tensorflow 2.0')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument('--datasets', nargs='+', help='Dataset directory path', required=True)
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net', default="mb1-ssd",
                    choices=['vgg16-ssd', 'mb1-ssd', 'mb1-ssd-lite', 'mb2-ssd-lite', 'sq-ssd-lite'],
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for Optimizer
parser.add_argument('--optimizer', default='SGD', type=str, help='What optimizer to use while training',
                    choices=['SGD', 'Adam', 'RAdam'])
parser.add_argument('--lr_scheduler', default='SGDR', choices=[None, 'SGDR'],
                    help="Use learning rate scheduler (only applies with SGD)")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optimizer')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=12, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--max_queue_size', default=10, type=int, help='Max number of batches to queue for training')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--save_format', choices=['h5', 'tf'],
                    help="What save format to use in model checkpoint callback. "
                         "h5 to save whole model to .h5 file and tf to save to Tensorflow SavedModel format",
                    default='h5')
parser.add_argument('--checkpoint_folder', default='models',
                    help='Directory for saving checkpoint models')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

if __name__ == '__main__':
    timer = Timer()
    logging.info(args)

    """
    Get net type
    """
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = create_mobilenetv2_ssd_lite
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    """
    Get Image augmentation classes
    """
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    """
    Prepare the training datasets
    """
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform, batch_size=args.batch_size,
                                 shuffle=True)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                        dataset_type="train", balance_data=args.balance_data,
                                        batch_size=args.batch_size, shuffle=True)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'tfrecord':
            dataset = RecordDataset(dataset_path, transform=train_transform, target_transform=target_transform,
                                    batch_size=args.batch_size, shuffle=True)
            label_file = os.path.join(args.checkpoint_folder, "tfrecord-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    datasets = ConcatDataset(datasets)
    logging.info(f"Stored labels into file {label_file}.")
    logging.info(f"Found {num_classes} classes")
    logging.info("Train dataset size: {}".format(datasets.data_length))

    """
    Get Validation dataset
    """
    logging.info("Prepare Validation dataset.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True,
                                 batch_size=args.batch_size, shuffle=False)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test", batch_size=args.batch_size, shuffle=False)
        logging.info(val_dataset)
    elif args.dataset_type == 'tfrecord':
        val_dataset = RecordDataset(args.validation_dataset, transform=test_transform,
                                    target_transform=target_transform,
                                    is_test=True, batch_size=args.batch_size,
                                    shuffle=False)
    else:
        logging.critical("Dataset type is unsupported. Choose on of voc, open_images, or tfrecord")
        raise SystemExit(-1)
    logging.info("validation dataset size: {}".format(val_dataset.num_records))

    """
    Build the network
    """
    logging.info("Build network.")
    timer.start("Create Model")
    net = create_net(num_classes, is_train=True)
    logging.info(f'Took {timer.end("Create Model"):.2f} seconds to create the model.')
    last_epoch = 0

    if args.freeze_base_net:
        logging.info("Freeze base net.")
        net.base_net.trainable = False
    elif args.freeze_net:
        for layer in net.ssd.layers:
            if layer not in net.regression_headers and layer not in net.classification_headers:
                layer.trainable = False
        logging.info("Freeze all the layers except prediction heads.")

    """
    Set up loss, optimizer, and callbacks
    """
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2)
    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(lr=args.lr, momentum=args.momentum, decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    elif args.optimizer == 'RAdam':
        os.environ['TF_KERAS'] = "1"
        from keras_radam import RAdam

        optimizer = RAdam(total_steps=len(datasets) * args.num_epochs,
                          weight_decay=args.weight_decay, amsgrad=False)
    else:
        logging.critical(f"Specified optimizer {args.optimizer} is unknown. Choose one of: SGD, Adam, RAdam")
        raise SystemExit(-1)

    """
    Load any specified weights before training
    """
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.ssd = tf.keras.models.load_model(args.resume,
                                             custom_objects={'forward': criterion.forward, args.optimizer: optimizer})
        epoch_idx = args.resume.find('Epoch')
        last_epoch = int(args.resume[epoch_idx + 6:epoch_idx + 8])
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.base_net.load_weights(args.base_net, by_name=True)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    """
    Set up callbacks
    """
    model_filename = str(args.net) + "-Epoch-{epoch:02d}-Loss-{val_loss:.2f}"
    if args.save_format == 'h5':
        model_filename += '.h5'

    callbacks = []
    checkpoint_path = os.path.join(args.checkpoint_folder, model_filename)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                     verbose=0,
                                                     mode='auto', baseline=None,
                                                     restore_best_weights=True)
    tensorboard = tf.keras.callbacks.TensorBoard()  # Not used, add to callbacks list to use
    plot = PlotLosses()
    if args.lr_scheduler == 'SGDR' and args.optimizer == 'SGD':
        lr_scheduler = SGDRScheduler(min_lr=1e-5, max_lr=1e-2,
                                     steps_per_epoch=len(datasets))
    callbacks.append(model_checkpoint)
    callbacks.append(early_stopper)
    callbacks.append(plot)

    logging.info(f"Learning rate: {args.lr}")

    # To test loss function
    # tf.keras.backend.set_learning_phase(0)
    # input, y_true = datasets[1]
    # with tf.device('/CPU:0'):
    #     y_pred = net.ssd.predict(input)
    # loss = criterion.forward(y_true, y_pred)

    """
    Begin training
    """
    net.ssd.compile(optimizer=optimizer, loss=criterion.forward, metrics=['accuracy'])
    logging.info(f"Start training from epoch {last_epoch}.")
    net.ssd.fit_generator(datasets,
                          steps_per_epoch=len(datasets),
                          epochs=args.num_epochs, verbose=1,
                          callbacks=callbacks, validation_data=val_dataset,
                          validation_steps=len(val_dataset), validation_freq=1,
                          initial_epoch=last_epoch, use_multiprocessing=False,
                          workers=args.num_workers, max_queue_size=args.max_queue_size)
