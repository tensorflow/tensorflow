# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for image preprocessing utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test

try:
  import PIL  # pylint:disable=g-import-not-at-top
except ImportError:
  PIL = None


def _generate_test_images():
  img_w = img_h = 20
  rgb_images = []
  gray_images = []
  for _ in range(8):
    bias = np.random.rand(img_w, img_h, 1) * 64
    variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
    imarray = np.random.rand(img_w, img_h, 3) * variance + bias
    im = keras.preprocessing.image.array_to_img(imarray, scale=False)
    rgb_images.append(im)

    imarray = np.random.rand(img_w, img_h, 1) * variance + bias
    im = keras.preprocessing.image.array_to_img(imarray, scale=False)
    gray_images.append(im)

  return [rgb_images, gray_images]


class TestImage(test.TestCase):

  def test_image_data_generator(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    for test_images in _generate_test_images():
      img_list = []
      for im in test_images:
        img_list.append(keras.preprocessing.image.img_to_array(im)[None, ...])

      images = np.vstack(img_list)
      generator = keras.preprocessing.image.ImageDataGenerator(
          featurewise_center=True,
          samplewise_center=True,
          featurewise_std_normalization=True,
          samplewise_std_normalization=True,
          zca_whitening=True,
          rotation_range=90.,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.5,
          zoom_range=0.2,
          channel_shift_range=0.,
          fill_mode='nearest',
          cval=0.5,
          horizontal_flip=True,
          vertical_flip=True)
      generator.fit(images, augment=True)

      for x, _ in generator.flow(
          images,
          np.arange(images.shape[0]),
          shuffle=True):
        self.assertEqual(x.shape[1:], images.shape[1:])
        break

  def test_image_data_generator_invalid_data(self):
    generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        data_format='channels_last')
    # Test fit with invalid data
    with self.assertRaises(ValueError):
      x = np.random.random((3, 10, 10))
      generator.fit(x)
    with self.assertRaises(ValueError):
      x = np.random.random((32, 3, 10, 10))
      generator.fit(x)
    with self.assertRaises(ValueError):
      x = np.random.random((32, 10, 10, 5))
      generator.fit(x)
    # Test flow with invalid data
    with self.assertRaises(ValueError):
      x = np.random.random((32, 10, 10, 5))
      generator.flow(np.arange(x.shape[0]))
    with self.assertRaises(ValueError):
      x = np.random.random((32, 10, 10))
      generator.flow(np.arange(x.shape[0]))
    with self.assertRaises(ValueError):
      x = np.random.random((32, 3, 10, 10))
      generator.flow(np.arange(x.shape[0]))

  def test_image_data_generator_fit(self):
    generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        data_format='channels_last')
    # Test grayscale
    x = np.random.random((32, 10, 10, 1))
    generator.fit(x)
    # Test RBG
    x = np.random.random((32, 10, 10, 3))
    generator.fit(x)
    generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        data_format='channels_first')
    # Test grayscale
    x = np.random.random((32, 1, 10, 10))
    generator.fit(x)
    # Test RBG
    x = np.random.random((32, 3, 10, 10))
    generator.fit(x)

  def test_directory_iterator(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    num_classes = 2

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)

    # create folders and subfolders
    paths = []
    for cl in range(num_classes):
      class_directory = 'class-{}'.format(cl)
      classpaths = [
          class_directory, os.path.join(class_directory, 'subfolder-1'),
          os.path.join(class_directory, 'subfolder-2'), os.path.join(
              class_directory, 'subfolder-1', 'sub-subfolder')
      ]
      for path in classpaths:
        os.mkdir(os.path.join(temp_dir, path))
      paths.append(classpaths)

    # save the images in the paths
    count = 0
    filenames = []
    for test_images in _generate_test_images():
      for im in test_images:
        # rotate image class
        im_class = count % num_classes
        # rotate subfolders
        classpaths = paths[im_class]
        filename = os.path.join(classpaths[count % len(classpaths)],
                                'image-{}.jpg'.format(count))
        filenames.append(filename)
        im.save(os.path.join(temp_dir, filename))
        count += 1

    # create iterator
    generator = keras.preprocessing.image.ImageDataGenerator()
    dir_iterator = generator.flow_from_directory(temp_dir)

    # check number of classes and images
    self.assertEqual(len(dir_iterator.class_indices), num_classes)
    self.assertEqual(len(dir_iterator.classes), count)
    self.assertEqual(sorted(dir_iterator.filenames), sorted(filenames))

  def test_img_utils(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    height, width = 10, 8

    # Test channels_first data format
    x = np.random.random((3, height, width))
    img = keras.preprocessing.image.array_to_img(
        x, data_format='channels_first')
    self.assertEqual(img.size, (width, height))
    x = keras.preprocessing.image.img_to_array(
        img, data_format='channels_first')
    self.assertEqual(x.shape, (3, height, width))
    # Test 2D
    x = np.random.random((1, height, width))
    img = keras.preprocessing.image.array_to_img(
        x, data_format='channels_first')
    self.assertEqual(img.size, (width, height))
    x = keras.preprocessing.image.img_to_array(
        img, data_format='channels_first')
    self.assertEqual(x.shape, (1, height, width))

    # Test channels_last data format
    x = np.random.random((height, width, 3))
    img = keras.preprocessing.image.array_to_img(x, data_format='channels_last')
    self.assertEqual(img.size, (width, height))
    x = keras.preprocessing.image.img_to_array(img, data_format='channels_last')
    self.assertEqual(x.shape, (height, width, 3))
    # Test 2D
    x = np.random.random((height, width, 1))
    img = keras.preprocessing.image.array_to_img(x, data_format='channels_last')
    self.assertEqual(img.size, (width, height))
    x = keras.preprocessing.image.img_to_array(img, data_format='channels_last')
    self.assertEqual(x.shape, (height, width, 1))


if __name__ == '__main__':
  test.main()
