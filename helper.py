import re
import random

import math
from math import pi

import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import PIL.Image


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if
                         not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def transform_image(image, shift, rotation, scale, flip):
    dtype_in = image.dtype  # Make sure we preserve dtype
    image = PIL.Image.fromarray(image)
    image = image.transform(
        image.size, PIL.Image.AFFINE,
        (scale, 0, shift[0], 0, scale, shift[1]))
    image.rotate(rotation)
    if flip:
        image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return np.array(image, dtype=dtype_in)

def random_transform(image, labelled_image):
    rotation = random.uniform(-10, 10)
    scale = 1.1 ** random.uniform(-1, 1)
    shift_x = random.uniform(-10, 10)
    shift_y = random.uniform(-10, 10)
    flip = random.choice([True, False])

    image, labelled_image = [
        transform_image(
            x,
            (shift_x, shift_y),
            rotation,
            scale,
            flip
        ) for x in (image, labelled_image)
    ]

    return image, labelled_image


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        path_list = glob(os.path.join(data_folder, 'image_2', '*.png'))
        labelled_image_map = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in
        glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        def load_training_images():
            for path in path_list:
                gt_image_file = labelled_image_map[os.path.basename(path)]

                image = scipy.misc.imresize(scipy.misc.imread(path),
                                            image_shape)
                labelled_image = scipy.misc.imresize(
                    scipy.misc.imread(gt_image_file),
                    image_shape)

                gt_bg = np.all(labelled_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                labelled_image = np.concatenate((gt_bg, np.invert(gt_bg)),
                                                axis=2)
                labelled_image = np.array(labelled_image, dtype=np.uint8)

                yield image, labelled_image

        def augment(sequence):
            for image, labelled_image in sequence:
                yield image, labelled_image
                yield random_transform(image, labelled_image)


        random.shuffle(path_list)
        image_list = []
        labelled_images_list = []
        for image, labelled_image in augment(load_training_images()):
            image_list.append(image)
            labelled_images_list.append(labelled_image)
            if len(image_list) == batch_size:
                yield np.array(image_list), np.array(labelled_images_list)
                image_list = []
                labelled_images_list = []

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder,
                    image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0],
                                                  image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                           keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image,
        os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
