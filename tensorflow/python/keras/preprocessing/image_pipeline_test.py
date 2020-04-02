# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image_pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.preprocessing import image as image_preproc
from tensorflow.python.keras.preprocessing import image_pipeline
from tensorflow.python.platform import test

try:
  import PIL  # pylint:disable=g-import-not-at-top
except ImportError:
  PIL = None


class DatasetFromDirectoryTest(keras_parameterized.TestCase):

  def _get_images(self, count=16, color_mode='rgb'):
    width = height = 24
    imgs = []
    for _ in range(count):
      if color_mode == 'grayscale':
        img = np.random.randint(0, 256, size=(height, width, 1))
      elif color_mode == 'rgba':
        img = np.random.randint(0, 256, size=(height, width, 4))
      else:
        img = np.random.randint(0, 256, size=(height, width, 3))
      img = image_preproc.array_to_img(img)
      imgs.append(img)
    return imgs

  def _prepare_directory(self,
                         num_classes=2,
                         grayscale=False,
                         nested_dirs=False,
                         color_mode='rgb',
                         count=16):
    # Get a unique temp directory
    temp_dir = os.path.join(self.get_temp_dir(), str(np.random.randint(1e6)))
    os.mkdir(temp_dir)
    self.addCleanup(shutil.rmtree, temp_dir)

    # Generate paths to class subdirectories
    paths = []
    for class_index in range(num_classes):
      class_directory = 'class_%s' % (class_index,)
      if nested_dirs:
        class_paths = [
            class_directory, os.path.join(class_directory, 'subfolder_1'),
            os.path.join(class_directory, 'subfolder_2'), os.path.join(
                class_directory, 'subfolder_1', 'sub-subfolder')
        ]
      else:
        class_paths = [class_directory]
      for path in class_paths:
        os.mkdir(os.path.join(temp_dir, path))
      paths += class_paths

    # Save images to the paths
    i = 0
    for img in self._get_images(color_mode=color_mode, count=count):
      path = paths[count % len(paths)]
      if color_mode == 'rgb':
        ext = 'jpg'
      else:
        ext = 'png'
      filename = os.path.join(path, 'image_%s.%s' % (i, ext))
      img.save(os.path.join(temp_dir, filename))
      i += 1
    return temp_dir

  def test_dataset_from_directory_binary(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=2)
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode='int')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    self.assertEqual(batch[0].dtype.name, 'float32')
    self.assertEqual(batch[1].shape, (8,))
    self.assertEqual(batch[1].dtype.name, 'int32')

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode='binary')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    self.assertEqual(batch[0].dtype.name, 'float32')
    self.assertEqual(batch[1].shape, (8, 1))
    self.assertEqual(batch[1].dtype.name, 'float32')

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode='categorical')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    self.assertEqual(batch[0].dtype.name, 'float32')
    self.assertEqual(batch[1].shape, (8, 2))
    self.assertEqual(batch[1].dtype.name, 'float32')

  def test_sample_count(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=4, count=15)
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode=None)
    sample_count = 0
    for batch in dataset:
      sample_count += batch.shape[0]
    self.assertEqual(sample_count, 15)

  def test_dataset_from_directory_multiclass(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=4, count=15)

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode=None)
    batch = next(iter(dataset))
    self.assertEqual(batch.shape, (8, 18, 18, 3))

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode=None)
    sample_count = 0
    iterator = iter(dataset)
    for batch in dataset:
      sample_count += next(iterator).shape[0]
    self.assertEqual(sample_count, 15)

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode='int')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    self.assertEqual(batch[0].dtype.name, 'float32')
    self.assertEqual(batch[1].shape, (8,))
    self.assertEqual(batch[1].dtype.name, 'int32')

    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode='categorical')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    self.assertEqual(batch[0].dtype.name, 'float32')
    self.assertEqual(batch[1].shape, (8, 4))
    self.assertEqual(batch[1].dtype.name, 'float32')

  def test_dataset_from_directory_color_modes(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=4, color_mode='rgba')
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), color_mode='rgba')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 4))
    self.assertEqual(batch[0].dtype.name, 'float32')

    directory = self._prepare_directory(num_classes=4, color_mode='grayscale')
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), color_mode='grayscale')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 1))
    self.assertEqual(batch[0].dtype.name, 'float32')

  def test_dataset_from_directory_validation_split(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=2, count=10)
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=10, image_size=(18, 18),
        validation_split=0.2, subset='training')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8, 18, 18, 3))
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=10, image_size=(18, 18),
        validation_split=0.2, subset='validation')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (2, 18, 18, 3))

  def test_dataset_from_directory_manual_labels(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=2, count=2)
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18),
        labels=[0, 1], shuffle=False)
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertAllClose(batch[1], [0, 1])

  def test_dataset_from_directory_follow_links(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=2, count=25,
                                        nested_dirs=True)
    dataset = image_pipeline.dataset_from_directory(
        directory, batch_size=8, image_size=(18, 18), label_mode=None,
        follow_links=True)
    sample_count = 0
    for batch in dataset:
      sample_count += batch.shape[0]
    self.assertEqual(sample_count, 25)

  def test_dataset_from_directory_errors(self):
    if PIL is None:
      return  # Skip test if PIL is not available.

    directory = self._prepare_directory(num_classes=3, count=5)

    with self.assertRaisesRegex(ValueError, '`labels` argument should be'):
      _ = image_pipeline.dataset_from_directory(
          directory, labels=None)

    with self.assertRaisesRegex(ValueError, '`label_mode` argument must be'):
      _ = image_pipeline.dataset_from_directory(directory, label_mode='other')

    with self.assertRaisesRegex(ValueError, '`color_mode` must be one of'):
      _ = image_pipeline.dataset_from_directory(directory, color_mode='other')

    with self.assertRaisesRegex(
        ValueError, 'only pass `class_names` if the labels are inferred'):
      _ = image_pipeline.dataset_from_directory(
          directory, labels=[0, 0, 1, 1, 1],
          class_names=['class_0', 'class_1', 'class_2'])

    with self.assertRaisesRegex(
        ValueError,
        'Expected the lengths of `labels` to match the number of images'):
      _ = image_pipeline.dataset_from_directory(directory, labels=[0, 0, 1, 1])

    with self.assertRaisesRegex(
        ValueError, '`class_names` passed did not match'):
      _ = image_pipeline.dataset_from_directory(
          directory, class_names=['class_0', 'class_2'])

    with self.assertRaisesRegex(ValueError, 'there must exactly 2 classes'):
      _ = image_pipeline.dataset_from_directory(directory, label_mode='binary')

    with self.assertRaisesRegex(ValueError,
                                '`validation_split` must be between 0 and 1'):
      _ = image_pipeline.dataset_from_directory(directory, validation_split=2)

    with self.assertRaisesRegex(ValueError,
                                '`subset` must be either "training" or'):
      _ = image_pipeline.dataset_from_directory(
          directory, validation_split=0.2, subset='other')


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
