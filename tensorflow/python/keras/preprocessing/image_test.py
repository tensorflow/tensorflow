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

import os
import shutil
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.preprocessing import image as preprocessing_image
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
    im = preprocessing_image.array_to_img(imarray, scale=False)
    rgb_images.append(im)

    imarray = np.random.rand(img_w, img_h, 1) * variance + bias
    im = preprocessing_image.array_to_img(imarray, scale=False)
    gray_images.append(im)

  return [rgb_images, gray_images]


class TestImage(keras_parameterized.TestCase):

  def test_smart_resize(self):
    test_input = np.random.random((20, 40, 3))
    output = preprocessing_image.smart_resize(test_input, size=(50, 50))
    self.assertIsInstance(output, np.ndarray)
    self.assertListEqual(list(output.shape), [50, 50, 3])
    output = preprocessing_image.smart_resize(test_input, size=(10, 10))
    self.assertListEqual(list(output.shape), [10, 10, 3])
    output = preprocessing_image.smart_resize(test_input, size=(100, 50))
    self.assertListEqual(list(output.shape), [100, 50, 3])
    output = preprocessing_image.smart_resize(test_input, size=(5, 15))
    self.assertListEqual(list(output.shape), [5, 15, 3])

  @parameterized.named_parameters(
      ('size1', (50, 50)),
      ('size2', (10, 10)),
      ('size3', (100, 50)),
      ('size4', (5, 15)))
  def test_smart_resize_tf_dataset(self, size):
    test_input_np = np.random.random((2, 20, 40, 3))
    test_ds = dataset_ops.Dataset.from_tensor_slices(test_input_np)

    resize = lambda img: preprocessing_image.smart_resize(img, size=size)
    test_ds = test_ds.map(resize)
    for sample in test_ds.as_numpy_iterator():
      self.assertIsInstance(sample, np.ndarray)
      self.assertListEqual(list(sample.shape), [size[0], size[1], 3])

  def test_smart_resize_batch(self):
    img = np.random.random((2, 20, 40, 3))
    out = preprocessing_image.smart_resize(img, size=(20, 20))
    self.assertListEqual(list(out.shape), [2, 20, 20, 3])
    self.assertAllClose(out, img[:, :, 10:-10, :])

  def test_smart_resize_errors(self):
    with self.assertRaisesRegex(ValueError, 'a tuple of 2 integers'):
      preprocessing_image.smart_resize(
          np.random.random((20, 20, 2)), size=(10, 5, 3))
    with self.assertRaisesRegex(ValueError, 'incorrect rank'):
      preprocessing_image.smart_resize(np.random.random((2, 4)), size=(10, 5))
    with self.assertRaisesRegex(ValueError, 'incorrect rank'):
      preprocessing_image.smart_resize(
          np.random.random((2, 4, 4, 5, 3)), size=(10, 5))

  def test_image_data_generator(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    for test_images in _generate_test_images():
      img_list = []
      for im in test_images:
        img_list.append(preprocessing_image.img_to_array(im)[None, ...])

      images = np.vstack(img_list)
      generator = preprocessing_image.ImageDataGenerator(
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
          brightness_range=(1, 5),
          fill_mode='nearest',
          cval=0.5,
          horizontal_flip=True,
          vertical_flip=True)
      # Basic test before fit
      x = np.random.random((32, 10, 10, 3))
      generator.flow(x)

      # Fit
      generator.fit(images, augment=True)

      for x, _ in generator.flow(
          images,
          np.arange(images.shape[0]),
          shuffle=True):
        self.assertEqual(x.shape[1:], images.shape[1:])
        break

  def test_image_data_generator_with_split_value_error(self):
    with self.assertRaises(ValueError):
      preprocessing_image.ImageDataGenerator(validation_split=5)

  def test_image_data_generator_invalid_data(self):
    generator = preprocessing_image.ImageDataGenerator(
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
    # Test flow with invalid data
    with self.assertRaises(ValueError):
      generator.flow(np.arange(5))
    # Invalid number of channels: will work but raise a warning
    x = np.random.random((32, 10, 10, 5))
    generator.flow(x)

    with self.assertRaises(ValueError):
      generator = preprocessing_image.ImageDataGenerator(
          data_format='unknown')

    generator = preprocessing_image.ImageDataGenerator(zoom_range=(2., 2.))

  def test_image_data_generator_fit(self):
    generator = preprocessing_image.ImageDataGenerator(
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
    generator = preprocessing_image.ImageDataGenerator(
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

    # Test image loading util
    fname = os.path.join(temp_dir, filenames[0])
    _ = preprocessing_image.load_img(fname)
    _ = preprocessing_image.load_img(fname, grayscale=True)
    _ = preprocessing_image.load_img(fname, target_size=(10, 10))
    _ = preprocessing_image.load_img(fname, target_size=(10, 10),
                                     interpolation='bilinear')

    # create iterator
    generator = preprocessing_image.ImageDataGenerator()
    dir_iterator = generator.flow_from_directory(temp_dir)

    # check number of classes and images
    self.assertEqual(len(dir_iterator.class_indices), num_classes)
    self.assertEqual(len(dir_iterator.classes), count)
    self.assertEqual(set(dir_iterator.filenames), set(filenames))

    def preprocessing_function(x):
      """This will fail if not provided by a Numpy array.

      Note: This is made to enforce backward compatibility.

      Args:
          x: A numpy array.

      Returns:
          An array of zeros with the same shape as the given array.
      """
      self.assertEqual(x.shape, (26, 26, 3))
      self.assertIs(type(x), np.ndarray)
      return np.zeros_like(x)

    # Test usage as Sequence
    generator = preprocessing_image.ImageDataGenerator(
        preprocessing_function=preprocessing_function)
    dir_seq = generator.flow_from_directory(
        str(temp_dir),
        target_size=(26, 26),
        color_mode='rgb',
        batch_size=3,
        class_mode='categorical')
    self.assertEqual(len(dir_seq), count // 3 + 1)
    x1, y1 = dir_seq[1]
    self.assertEqual(x1.shape, (3, 26, 26, 3))
    self.assertEqual(y1.shape, (3, num_classes))
    x1, y1 = dir_seq[5]
    self.assertTrue((x1 == 0).all())

  def directory_iterator_with_validation_split_test_helper(
      self, validation_split):
    if PIL is None:
      return  # Skip test if PIL is not available.

    num_classes = 2
    tmp_folder = tempfile.mkdtemp(prefix='test_images')

    # create folders and subfolders
    paths = []
    for cl in range(num_classes):
      class_directory = 'class-{}'.format(cl)
      classpaths = [
          class_directory,
          os.path.join(class_directory, 'subfolder-1'),
          os.path.join(class_directory, 'subfolder-2'),
          os.path.join(class_directory, 'subfolder-1', 'sub-subfolder')
      ]
      for path in classpaths:
        os.mkdir(os.path.join(tmp_folder, path))
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
        im.save(os.path.join(tmp_folder, filename))
        count += 1

    # create iterator
    generator = preprocessing_image.ImageDataGenerator(
        validation_split=validation_split)

    with self.assertRaises(ValueError):
      generator.flow_from_directory(tmp_folder, subset='foo')

    num_validation = int(count * validation_split)
    num_training = count - num_validation
    train_iterator = generator.flow_from_directory(
        tmp_folder, subset='training')
    self.assertEqual(train_iterator.samples, num_training)

    valid_iterator = generator.flow_from_directory(
        tmp_folder, subset='validation')
    self.assertEqual(valid_iterator.samples, num_validation)

    # check number of classes and images
    self.assertEqual(len(train_iterator.class_indices), num_classes)
    self.assertEqual(len(train_iterator.classes), num_training)
    self.assertEqual(
        len(set(train_iterator.filenames) & set(filenames)), num_training)

    model = sequential.Sequential([layers.Flatten(), layers.Dense(2)])
    model.compile(optimizer='sgd', loss='mse')
    model.fit(train_iterator, epochs=1)

    shutil.rmtree(tmp_folder)

  @keras_parameterized.run_all_keras_modes
  def test_directory_iterator_with_validation_split_25_percent(self):
    self.directory_iterator_with_validation_split_test_helper(0.25)

  @keras_parameterized.run_all_keras_modes
  def test_directory_iterator_with_validation_split_40_percent(self):
    self.directory_iterator_with_validation_split_test_helper(0.40)

  @keras_parameterized.run_all_keras_modes
  def test_directory_iterator_with_validation_split_50_percent(self):
    self.directory_iterator_with_validation_split_test_helper(0.50)

  def test_img_utils(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    height, width = 10, 8

    # Test channels_first data format
    x = np.random.random((3, height, width))
    img = preprocessing_image.array_to_img(
        x, data_format='channels_first')
    self.assertEqual(img.size, (width, height))
    x = preprocessing_image.img_to_array(
        img, data_format='channels_first')
    self.assertEqual(x.shape, (3, height, width))
    # Test 2D
    x = np.random.random((1, height, width))
    img = preprocessing_image.array_to_img(
        x, data_format='channels_first')
    self.assertEqual(img.size, (width, height))
    x = preprocessing_image.img_to_array(
        img, data_format='channels_first')
    self.assertEqual(x.shape, (1, height, width))

    # Test channels_last data format
    x = np.random.random((height, width, 3))
    img = preprocessing_image.array_to_img(x, data_format='channels_last')
    self.assertEqual(img.size, (width, height))
    x = preprocessing_image.img_to_array(img, data_format='channels_last')
    self.assertEqual(x.shape, (height, width, 3))
    # Test 2D
    x = np.random.random((height, width, 1))
    img = preprocessing_image.array_to_img(x, data_format='channels_last')
    self.assertEqual(img.size, (width, height))
    x = preprocessing_image.img_to_array(img, data_format='channels_last')
    self.assertEqual(x.shape, (height, width, 1))

  def test_batch_standardize(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    # ImageDataGenerator.standardize should work on batches
    for test_images in _generate_test_images():
      img_list = []
      for im in test_images:
        img_list.append(preprocessing_image.img_to_array(im)[None, ...])

      images = np.vstack(img_list)
      generator = preprocessing_image.ImageDataGenerator(
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
          brightness_range=(1, 5),
          fill_mode='nearest',
          cval=0.5,
          horizontal_flip=True,
          vertical_flip=True)
      generator.fit(images, augment=True)

      transformed = np.copy(images)
      for i, im in enumerate(transformed):
        transformed[i] = generator.random_transform(im)
      transformed = generator.standardize(transformed)

  def test_img_transforms(self):
    x = np.random.random((3, 200, 200))
    _ = preprocessing_image.random_rotation(x, 20)
    _ = preprocessing_image.random_shift(x, 0.2, 0.2)
    _ = preprocessing_image.random_shear(x, 2.)
    _ = preprocessing_image.random_zoom(x, (0.5, 0.5))
    _ = preprocessing_image.apply_channel_shift(x, 2, 2)
    _ = preprocessing_image.apply_affine_transform(x, 2)
    with self.assertRaises(ValueError):
      preprocessing_image.random_zoom(x, (0, 0, 0))
    _ = preprocessing_image.random_channel_shift(x, 2.)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
