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
"""Tests for text_dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
import string

from tensorflow.python.compat import v2_compat
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.preprocessing import text_dataset
from tensorflow.python.platform import test


class TextDatasetFromDirectoryTest(keras_parameterized.TestCase):

  def _prepare_directory(self,
                         num_classes=2,
                         nested_dirs=False,
                         count=16,
                         length=20):
    # Get a unique temp directory
    temp_dir = os.path.join(self.get_temp_dir(), str(random.randint(0, 1e6)))
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

    for i in range(count):
      path = paths[count % len(paths)]
      filename = os.path.join(path, 'text_%s.txt' % (i,))
      f = open(os.path.join(temp_dir, filename), 'w')
      text = ''.join([random.choice(string.printable) for _ in range(length)])
      f.write(text)
      f.close()
    return temp_dir

  def test_text_dataset_from_directory_binary(self):
    directory = self._prepare_directory(num_classes=2)
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode='int', max_length=10)
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    self.assertEqual(batch[0].dtype.name, 'string')
    self.assertEqual(len(batch[0].numpy()[0]), 10)  # Test max_length
    self.assertEqual(batch[1].shape, (8,))
    self.assertEqual(batch[1].dtype.name, 'int32')

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode='binary')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    self.assertEqual(batch[0].dtype.name, 'string')
    self.assertEqual(batch[1].shape, (8, 1))
    self.assertEqual(batch[1].dtype.name, 'float32')

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode='categorical')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    self.assertEqual(batch[0].dtype.name, 'string')
    self.assertEqual(batch[1].shape, (8, 2))
    self.assertEqual(batch[1].dtype.name, 'float32')

  def test_sample_count(self):
    directory = self._prepare_directory(num_classes=4, count=15)
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode=None)
    sample_count = 0
    for batch in dataset:
      sample_count += batch.shape[0]
    self.assertEqual(sample_count, 15)

  def test_text_dataset_from_directory_multiclass(self):
    directory = self._prepare_directory(num_classes=4, count=15)

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode=None)
    batch = next(iter(dataset))
    self.assertEqual(batch.shape, (8,))

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode=None)
    sample_count = 0
    iterator = iter(dataset)
    for batch in dataset:
      sample_count += next(iterator).shape[0]
    self.assertEqual(sample_count, 15)

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode='int')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    self.assertEqual(batch[0].dtype.name, 'string')
    self.assertEqual(batch[1].shape, (8,))
    self.assertEqual(batch[1].dtype.name, 'int32')

    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode='categorical')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    self.assertEqual(batch[0].dtype.name, 'string')
    self.assertEqual(batch[1].shape, (8, 4))
    self.assertEqual(batch[1].dtype.name, 'float32')

  def test_text_dataset_from_directory_validation_split(self):
    directory = self._prepare_directory(num_classes=2, count=10)
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=10, validation_split=0.2, subset='training')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (8,))
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=10, validation_split=0.2, subset='validation')
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertEqual(batch[0].shape, (2,))

  def test_text_dataset_from_directory_manual_labels(self):
    directory = self._prepare_directory(num_classes=2, count=2)
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, labels=[0, 1], shuffle=False)
    batch = next(iter(dataset))
    self.assertLen(batch, 2)
    self.assertAllClose(batch[1], [0, 1])

  def test_text_dataset_from_directory_follow_links(self):
    directory = self._prepare_directory(num_classes=2, count=25,
                                        nested_dirs=True)
    dataset = text_dataset.text_dataset_from_directory(
        directory, batch_size=8, label_mode=None, follow_links=True)
    sample_count = 0
    for batch in dataset:
      sample_count += batch.shape[0]
    self.assertEqual(sample_count, 25)

  def test_text_dataset_from_directory_errors(self):
    directory = self._prepare_directory(num_classes=3, count=5)

    with self.assertRaisesRegex(ValueError, '`labels` argument should be'):
      _ = text_dataset.text_dataset_from_directory(
          directory, labels=None)

    with self.assertRaisesRegex(ValueError, '`label_mode` argument must be'):
      _ = text_dataset.text_dataset_from_directory(
          directory, label_mode='other')

    with self.assertRaisesRegex(
        ValueError, 'only pass `class_names` if the labels are inferred'):
      _ = text_dataset.text_dataset_from_directory(
          directory, labels=[0, 0, 1, 1, 1],
          class_names=['class_0', 'class_1', 'class_2'])

    with self.assertRaisesRegex(
        ValueError,
        'Expected the lengths of `labels` to match the number of files'):
      _ = text_dataset.text_dataset_from_directory(
          directory, labels=[0, 0, 1, 1])

    with self.assertRaisesRegex(
        ValueError, '`class_names` passed did not match'):
      _ = text_dataset.text_dataset_from_directory(
          directory, class_names=['class_0', 'class_2'])

    with self.assertRaisesRegex(ValueError, 'there must exactly 2 classes'):
      _ = text_dataset.text_dataset_from_directory(
          directory, label_mode='binary')

    with self.assertRaisesRegex(ValueError,
                                '`validation_split` must be between 0 and 1'):
      _ = text_dataset.text_dataset_from_directory(
          directory, validation_split=2)

    with self.assertRaisesRegex(ValueError,
                                '`subset` must be either "training" or'):
      _ = text_dataset.text_dataset_from_directory(
          directory, validation_split=0.2, subset='other')


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
