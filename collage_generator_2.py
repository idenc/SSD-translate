import collections
import io
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import Augmentor
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw


def main():
    """
    Modify the CONFIGS dictionary to your desired directories/settings
    """
    CONFIGS = {
        # Maximum number of masked image to paste over background
        'max_num_samples': 3,
        # Path to masked images
        'dataset_path': r'D:\Fruit-Images-Dataset-master\Fruit-Images-Dataset-master\Training',
        # Path to save JPEG images to, only applies if save_to_jpg is True
        'collage_save_path': r"C:\Users\iden\Documents\collages",
        # Path to save collage TFRecord files to
        'collage_record_save_path': r"D:\fruits_large",
        # Whether to save JPG images of each generated collage
        'save_to_jpg': False,
        # Number of collages to generate
        'collage_count': 100,
        # Train-validation split ratio
        'train_split': 0.97,
        # How many collages to save in a single TFRecord file
        # This is to enable true shuffling during training as TFRecord files are sequential only
        'shard_size': 500,
        # Settings for augmentor library augmentations to apply to masked images
        'augmentation_settings': {
            'rotate': {
                'probability': 0.5,
                'max_left_rotation': 10,
                'max_right_rotation': 10,
            },
            'flip_left_right': {
                'probability': 0.5,
            },
            'flip_top_bottom': {
                'probability': 0.3,
            },
            'random_distortion': {
                'probability': 0.5,
                'grid_width': 3,
                'grid_height': 3,
                'magnitude': 8
            },
            'rotate_random_90': {
                'probability': 0.5
            },
            'skew': {
                'probability': 0.5
            },
            'random_contrast': {
                'probability': 0.4,
                'min_factor': 0.5,
                'max_factor': 2.0
            },
            'random_brightness': {
                'probability': 0.4,
                'min_factor': 0.5,
                'max_factor': 2.0
            }
        },
        'num_workers': 12,  # Number of threads to use when generating images
        # Probability to exclude sample from collage
        # A single collage will always have at least one sample
        'mask_probability': 0.4,
        # Applies flood fill to each sample image to mask solid background out
        'use_floodfill': True,
        'collage_background_type': 'img',  # `img` for background images, `solid` for solid colour background
        # Path to background images to use
        'collage_background_path': r"D:\VOC_datasets\VOC2012\JPEGImages",
        # Random samples from background directory
        'collage_background_colour': (255, 255, 255)
    }

    class Sample:
        """
            Wrapper for a singly-labelled image stored as a PIL Image.

            Attributes
            ----------
            image_path: string
                absolute/relative path to image
            cls: string
                class label corresponding to image
            image: PIL.Image
                image data stored as a PIL Image
            bounding_box: list(int)
                list of four integers specifying bounding box. *Note that since
                these are expected to be unannotated images, the image boundaries
                are considered as its bounding box. So in effect, for all
                unpositioned samples, bounding boxes are expected to be:

                    [0, 0, <width>, <height>], and

                for positioned samples (within a collage), bounding boxes are expected
                to be:

                    [<x>, <y>, <width>, <height>]

                where <x> & <y> are the coordinates of the top-left corner of the sample
                within the collage.

            Methods
            -------
            display()
                uses matplotlib to display the PIL image

        """

        def __init__(self, image_path, cls, image):
            self.image_path = image_path
            self.cls = cls
            self.image = image
            self.width, self.height = image.size

        def display(self):
            plt.imshow(self.image)

    dataset, dataset_path = {}, CONFIGS['dataset_path']

    assert os.path.exists(dataset_path), "ERROR: dataset path is invalid."
    classes = [cls for cls in os.listdir(dataset_path) if not cls.startswith('.')]
    assert (classes is not None) or (len(classes) != 0), "ERROR: empty dataset directory."

    # create a list of images in each class directory
    for cls in classes:
        files = [file for file in os.listdir(os.path.join(dataset_path, cls)) if not file.startswith('.')]
        if (files is not None) or (len(files) != 0):
            file_paths = [os.path.join(dataset_path, cls, file) for file in files]
            images = []
            for img_path in file_paths:
                # images.append((img_path, Image.open(img_path).convert("RGBA")))
                images.append(img_path)
            dataset[cls] = images
        else:
            print("WARNING: class directory '" + cls + "' is empty -> skipping.")

    if CONFIGS['collage_background_type'] == 'img':  # Use random image as background
        backgrounds = []
        background_paths = [os.path.join(CONFIGS['collage_background_path'], f)
                            for f in os.listdir(CONFIGS['collage_background_path'])]
        for path in background_paths:
            # backgrounds.append(Image.open(path).convert("RGB"))
            backgrounds.append(path)

    # Collage Definition
    class SquareCollage:
        """
            A grid-like square 'collage' of singly-labelled images stored as a
            PIL Image.

            Attributes
            ----------
            CONFIGS: dict
                a dictionary specifying various collage configurations. The following keys
                are expected:
                                    KEY | DESCRIPTION
                --------------------------------------------------------------------------
                 'side_length_in_tiles' | # of samples to place along each dimension
                'side_length_in_pixels' | total pixel side length of collage
                         'dataset_path' | path to class-organized dataset root directory
                'augmentation_settings' | dictionary of augmentation settings**
                     'mask_probability' | probability with which each sample is masked out
                --------------------------------------------------------------------------
                ** supported augmentations:

                    - 'rotate'
                    - 'zoom'
                    - 'flip_left_right'
                    - 'flip_top_bottom'

                   Configurations for each of these is expected to be a dictionary that is
                   unpackable as arguments into corresponding Augmentor library functions.
                   Refer to the following link to see which arguments are valid:
                   https://augmentor.readthedocs.io/en/master/code.html#module-Augmentor.Operations

            original_samples: 2D numpy array of Samples
                a 2D numpy array of Sample objects containing original image data
            augmented_samples: 2D numpy array of Samples
                a 2D numpy array of Sample objects containing augmented image data
            mask: numpy.array(numpy.array(boolean))
                a 2D numpy array of booleans (matching dimensions of sample arrays)
            collage: PIL.Image
                a PIL Image comprised of masked augmented samples
            bounding_boxes: 2D list
                list of bounding boxes corresponding to the objects in the collage

            Methods
            -------
            __sample_from_dataset()
            __create_augmented_samples()
            __create_collage()
            display()
            save(save_path)
            generate_new_augmented_samples()
            generate_new_mask(masking_probability)

        """

        def __init__(self, CONFIGS, print_attributes=False):

            self.CONFIGS = CONFIGS

            self.__sample_from_dataset()
            self.__create_augmented_samples()
            self.__create_collage()

            if print_attributes: print(vars(self))

        ###################### PRIVATE METHODS ######################

        def __sample_from_dataset(self):

            """
                Creates a per-class list of all images in specified dataset and randomly
                samples N^2 images to load as PIL images (where N = required # of tiles
                along each dimension of collage).

                Parameters/Returns
                ------------------
                None

                Notes
                -----
                * This method asserts that sampled images are unique.

            """

            samples_required = self.CONFIGS['max_num_samples']

            samples, sample_paths = [], []

            # sample required number of samples from dataset image lists
            while samples_required > 0:

                # randomly select a class
                random_class = random.choice(classes)

                # randomly select an image from the class
                random_image_path = random.choice(dataset[random_class])

                # assert uniqueness then save as a Sample object
                if random_image_path not in sample_paths:
                    sample_paths.append(random_image_path)
                    sample = Image.open(random_image_path).convert("RGBA")

                    sample = Sample(random_image_path,
                                    random_class,
                                    sample)
                    samples.append(sample)
                    samples_required -= 1

            self.original_samples = samples

        def __create_augmented_samples(self):

            """
                Creates augmented versions of original sampled images using specified
                augmentation settings.

                Parameters/Returns
                -----------------
                None

            """
            self.pipeline = Augmentor.Pipeline()

            augmentation_settings = self.CONFIGS['augmentation_settings']
            # Add each of the sample augmentations to pipeline with desired settings
            self.pipeline.rotate_without_crop(**augmentation_settings['rotate'])
            self.pipeline.random_distortion(**augmentation_settings['random_distortion'])
            self.pipeline.flip_left_right(**augmentation_settings['flip_left_right'])
            self.pipeline.flip_top_bottom(**augmentation_settings['flip_top_bottom'])
            self.pipeline.rotate_random_90(**augmentation_settings['rotate_random_90'])
            # self.pipeline.skew(**augmentation_settings['skew'])
            self.pipeline.random_contrast(**augmentation_settings['random_contrast'])
            self.pipeline.random_brightness(**augmentation_settings['random_brightness'])

        def __paste_image(self, collage):
            """
            Pastes masked images onto background with augmentations
            :param collage:
            :return:
            """
            bounding_boxes, labels = [], []
            max_num_samples = self.CONFIGS['max_num_samples']

            # Determine how many samples will be present in this image
            show_tile = [False for _ in range(max_num_samples)]
            num_shown_samples = 0
            for i in range(max_num_samples):
                mask_prob = round(random.uniform(0, 1), 1)
                if mask_prob > self.CONFIGS['mask_probability']:
                    show_tile[i] = True
                    num_shown_samples += 1
            if num_shown_samples == 0:
                num_shown_samples = 1

            # Divide up image into sections to avoid samples overlapping each other
            # region_coords[i] consists of (xmin, ymin, xmax, ymax) for sample i
            region_coords = []
            if collage.height > collage.width:  # Image is taller than it is wide
                region_height = collage.height // num_shown_samples
                for i in range(num_shown_samples):
                    region_coords.append((0, region_height * i, collage.width, region_height * (i + 1)))
            else:  # Image is square or wider than it is tall
                region_width = collage.width // num_shown_samples
                for i in range(num_shown_samples):
                    region_coords.append((region_width * i, 0, region_width * (i + 1), collage.height))

            # loop through all samples & mask, resize, and paste onto empty collage accordingly
            for i in range(max_num_samples):
                # determine whether this tile is to be masked
                if show_tile[i] or (i == (max_num_samples - 1) and not bounding_boxes):
                    # extract tile information
                    tile = self.original_samples[i]
                    image_to_paste = tile.image
                    if self.CONFIGS['use_floodfill']:
                        # Fill background with transparency
                        ImageDraw.floodfill(image_to_paste, (0, 0), (255, 255, 255, 0), thresh=90)

                    # Get region coords
                    sample_coords = region_coords.pop(0)
                    xmin = sample_coords[0]
                    ymin = sample_coords[1]
                    xmax = sample_coords[2]
                    ymax = sample_coords[3]
                    region_width = xmax - xmin
                    region_height = ymax - ymin

                    # Resize sample to fit region
                    if region_width > region_height:
                        # resize image
                        scale = region_width / region_height
                        # sample takes up minimum 30% of background image and maximum 95%
                        new_size = random.randint(int(region_height * 0.3), int(region_height * 0.95))
                        new_size = (int(new_size * scale), new_size)
                    else:
                        # resize image
                        scale = region_height / region_width
                        # sample takes up minimum 30% of background image and maximum 95%
                        new_size = random.randint(int(region_width * 0.3), int(region_width * 0.95))
                        new_size = (new_size, int(new_size * scale))

                    image_to_paste = image_to_paste.resize(new_size, Image.ANTIALIAS)

                    # Perform augmentations based on probability
                    for operation in self.pipeline.operations:
                        r = round(random.uniform(0, 1), 1)
                        if r <= operation.probability:
                            image_to_paste = operation.perform_operation([image_to_paste])[0]

                    # image_to_paste = Image.fromarray(noisy('poisson', image_to_paste))
                    # determine position of image within collage
                    x = random.randint(xmin, max(xmax - image_to_paste.width, xmin + 1))
                    if x >= collage.width - image_to_paste.width:
                        x = collage.width - image_to_paste.width
                    y = random.randint(ymin, max(ymax - image_to_paste.height, ymin + 1))
                    if y >= collage.height - image_to_paste.height:
                        y = collage.height - image_to_paste.height

                    # determine sample's bounding box and ensure bounding box is not outside image
                    # coords have format [min_x, min_y, max_x, max_y]
                    bounding_box = [x, y,
                                    min(x + image_to_paste.width, collage.width),
                                    min(y + image_to_paste.height, collage.height)]
                    # paste image onto collage
                    collage.paste(image_to_paste, (x, y), mask=image_to_paste)

                    # save corresponding bounding box & label
                    bounding_boxes.append(bounding_box)
                    labels.append(tile.cls)

            return bounding_boxes, labels

        def __create_collage(self):
            """
            Creates a square collage using the augmented samples and mask generated prior to
            this method call and saves corresponding bounding boxes and labels in order.

            Parameters/Returns
            ------------------
            None

            """
            # create an empty collage image
            if self.CONFIGS['collage_background_type'] == 'img':  # Use random image as background
                collage = Image.open(random.choice(backgrounds))
            else:  # Use a default solid background
                background_colour = self.CONFIGS['collage_background_colour']
                collage = Image.new('RGB',
                                    (900, 900),
                                    color=background_colour)

            bounding_boxes, labels = self.__paste_image(collage)

            self.collage = collage
            self.bounding_boxes = bounding_boxes
            self.labels = labels

        ###################### PUBLIC METHODS ######################

        def display(self, display_bounding_boxes=False):
            """
            Draws collage with option to display bounding boxes
            :param display_bounding_boxes: Whether to display bounding boxes
            :return:
            """
            if display_bounding_boxes:
                draw = ImageDraw.Draw(self.collage)
                for box in self.bounding_boxes:
                    # Draw bounding boxes
                    draw.rectangle(box, outline=(0, 255, 0), width=5)
            plt.imshow(self.collage)

        def save(self, save_path):
            pass
            # self.collage.save(save_path)

        def generate_new_augmented_samples(self):
            self.__create_augmented_samples()

    # Create TF Example

    def create_tf_example(annotation, class_label_map, img):
        """
        Creates tf example to be written to TFRecord
        :param annotation: sample annotation data
        :param class_label_map: class label dictionary
        :param img: collage image
        :return: a TF example containing sample data
        """
        # load image data
        filename = annotation['filename'].encode('utf8')
        image_format = b'jpg'

        output = io.BytesIO()
        img.save(output, format='JPEG')
        encoded_jpg = output.getvalue()
        width, height = annotation['width'], annotation['height']

        # check if the image format is matching with your images.
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = ([] for i in range(6))

        for bounding_box in annotation['boxes']:
            # append normalized bounding box values
            xmins.append(bounding_box['left'] / width)
            xmaxs.append(bounding_box['right'] / width)
            ymins.append(bounding_box['top'] / height)
            ymaxs.append(bounding_box['bottom'] / height)

            classes_text.append(bounding_box['class_name'].encode('utf8'))
            classes.append(class_label_map[bounding_box['class_name']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        }))

        return tf_example

    def create_annotation(collage, filename):
        """
        Creates annotation to write to TFRecord file
        :param collage: Data point
        :param filename: Filename of data point
        :return:
        """
        annotation = {'filename': filename, 'width': collage.collage.size[0], 'height': collage.collage.size[1],
                      'boxes': []}

        for box, label in zip(collage.bounding_boxes, collage.labels):
            bbox = {'class_name': label, 'left': box[0], 'top': box[1], 'right': box[2], 'bottom': box[3]}
            annotation['boxes'].append(bbox)

        return annotation

    def save_label_map(label_map, output_path):
        """
        Saves class labels in TFRecord format
        :param label_map: Label map dictionary
        :param output_path: Path to write label_map.txt
        :return:
        """
        with open(os.path.join(output_path, "label_map.txt"), "w") as f:
            for label in list(label_map.keys()):
                line = "item {\n    id: " + str(label_map[label]) + "\n    name: '" + label + "'\n}\n"
                f.write(line)

    def gen_collages(t, train_count, val_count, train_writers, val_writer, class_label_map):
        """
        Function to generate collages and write to TFRecord/JPEG
        Function is called in parallel to speed up generation
        :param t: Index of collage
        :param train_count: Size of training set
        :param val_count: Size of validation set
        :param train_writers: TFRecord writers for train set
        :param val_writer: TFRecord writers for validation set
        :param class_label_map: List containing class labels
        """
        collage = SquareCollage(CONFIGS)

        if t < train_count:
            if (t + 1) % 200 == 0:
                print("processing train set annotations " + str(t + 1) + " of " + str(train_count))
            filename = "aug_train_" + str(t) + ".jpg"
            writer = train_writers[t // CONFIGS['shard_size']]
        else:
            if (t + 1) % 200 == 0:
                print("processing val set annotations " + str(t - train_count + 1) + " of " + str(val_count))
            filename = "aug_val_" + str(t) + ".jpg"
            writer = val_writer

        # save collage as JPEG image
        if CONFIGS['save_to_jpg']:
            image_save_path = os.path.join(CONFIGS['collage_save_path'], filename)
            collage.collage.save(image_save_path, "JPEG")

        # create a TF example using image + annotation data
        annotation = create_annotation(collage, filename)
        val_tf_example = create_tf_example(annotation, class_label_map, collage.collage)

        # write serialized TF example
        writer.write(val_tf_example.SerializeToString())

    def create_collages():
        """
        Entry point to generate collage
        :return:
        """
        # create save directories
        os.makedirs(CONFIGS['collage_save_path'], exist_ok=True)
        os.makedirs(CONFIGS['collage_record_save_path'], exist_ok=True)

        # generate required number of collages
        train_count = int(CONFIGS['train_split'] * CONFIGS['collage_count'])
        val_count = CONFIGS['collage_count'] - train_count

        # create label map
        classes = [class_label for class_label in set(os.listdir(CONFIGS['dataset_path'])) if
                   not class_label.startswith('.')]
        # Sort to ensure labels are in the same order every time data is generated
        classes.sort()
        class_label_map = collections.OrderedDict()

        # create train & val set writers
        base_path = os.path.join(CONFIGS['collage_record_save_path'], "train")
        num_shards = math.ceil(train_count / CONFIGS['shard_size'])
        tf_record_output_filenames = [
            '{}-{:05d}-of-{:05d}.record'.format(base_path, idx, num_shards)
            for idx in range(num_shards)
        ]
        train_writers = [tf.io.TFRecordWriter(file_name) for file_name in tf_record_output_filenames]
        val_writer = tf.io.TFRecordWriter(os.path.join(CONFIGS['collage_record_save_path'], "val.record"))

        for k in range(len(classes)):
            class_label_map[classes[k]] = k + 1

        save_label_map(class_label_map, CONFIGS['collage_record_save_path'])
        print("Generating collage...")
        with ThreadPoolExecutor(CONFIGS['num_workers']) as executor:
            futures = [executor.submit(gen_collages,
                                       t, train_count, val_count, train_writers, val_writer, class_label_map)
                       for t in range(CONFIGS['collage_count'])]
            for future in futures:
                future.result()

    # Finally create all collages
    create_collages()


if __name__ == '__main__':
    main()
