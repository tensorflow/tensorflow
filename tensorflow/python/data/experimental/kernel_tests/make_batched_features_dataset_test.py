# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.make_batched_features_dataset()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class MakeBatchedFeaturesDatasetTest(tf_record_test_base.FeaturesTestBase,
                                     parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.combine(batch_size=[1, 2], num_epochs=[1, 10]),
          test_base.default_test_combinations()))
  def testRead(self, batch_size, num_epochs):
    # Basic test: read from file 0.
    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[0],
            label_key="label",
            num_epochs=num_epochs,
            batch_size=batch_size))
    self._verify_records(
        batch_size, 0, num_epochs=num_epochs, label_key_provided=True)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch(label_key_provided=True)

      # Basic test: read from file 1.
    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[1],
            label_key="label",
            num_epochs=num_epochs,
            batch_size=batch_size))
    self._verify_records(
        batch_size, 1, num_epochs=num_epochs, label_key_provided=True)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch(label_key_provided=True)

    # Basic test: read from both files.
    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames,
            label_key="label",
            num_epochs=num_epochs,
            batch_size=batch_size))
    self._verify_records(
        batch_size, num_epochs=num_epochs, label_key_provided=True)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch(label_key_provided=True)
    # Basic test: read from both files.
    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames,
            num_epochs=num_epochs,
            batch_size=batch_size))
    self._verify_records(batch_size, num_epochs=num_epochs)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch()

  @combinations.generate(test_base.default_test_combinations())
  def testReadWithEquivalentDataset(self):
    features = {
        "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        "record": parsing_ops.FixedLenFeature([], dtypes.int64),
    }
    dataset = (
        core_readers.TFRecordDataset(self._filenames)
        .map(lambda x: parsing_ops.parse_single_example(x, features))
        .repeat(10).batch(2))
    next_element = self.getNext(dataset)
    for file_batch, _, _, _, record_batch, _ in self._next_expected_batch(
        range(self._num_files), 2, 10):
      actual_batch = self.evaluate(next_element())
      self.assertAllEqual(file_batch, actual_batch["file"])
      self.assertAllEqual(record_batch, actual_batch["record"])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(
      combinations.times(
          combinations.combine(batch_size=[1, 2], num_epochs=[5]),
          test_base.default_test_combinations()))
  def testReadWithFusedShuffleRepeatDatasetSameSeed(self, batch_size,
                                                    num_epochs):
    total_records = num_epochs * self._num_records
    # Test that shuffling with same seed produces the same result.
    outputs1 = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[0],
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            shuffle_seed=5))
    outputs2 = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[0],
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            shuffle_seed=5))
    for _ in range(total_records // batch_size):
      batch1 = self._run_actual_batch(outputs1)
      batch2 = self._run_actual_batch(outputs2)
      for i in range(len(batch1)):
        self.assertAllEqual(batch1[i], batch2[i])

  @combinations.generate(
      combinations.times(
          combinations.combine(batch_size=[1, 2], num_epochs=[5]),
          test_base.default_test_combinations()))
  def testReadWithFusedShuffleRepeatDatasetDifferentSeed(
      self, batch_size, num_epochs):
    total_records = num_epochs * self._num_records
    # Test that shuffling with different seeds produces a different order.
    outputs1 = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[0],
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            shuffle_seed=5))
    outputs2 = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames[0],
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            shuffle_seed=15))
    all_equal = True
    for _ in range(total_records // batch_size):
      batch1 = self._run_actual_batch(outputs1)
      batch2 = self._run_actual_batch(outputs2)
      for i in range(len(batch1)):
        all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
    self.assertFalse(all_equal)

  @combinations.generate(
      combinations.times(
          combinations.combine(
              batch_size=[1, 2],
              num_epochs=[5],
              reader_num_threads=[2, 4],
              parser_num_threads=[2, 4]),
          test_base.default_test_combinations()))
  def testParallelReadersAndParsers(self, batch_size, num_epochs,
                                    reader_num_threads, parser_num_threads):
    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames,
            label_key="label",
            num_epochs=num_epochs,
            batch_size=batch_size,
            reader_num_threads=reader_num_threads,
            parser_num_threads=parser_num_threads))
    self._verify_records(
        batch_size,
        num_epochs=num_epochs,
        label_key_provided=True,
        interleave_cycle_length=reader_num_threads)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch(label_key_provided=True)

    self.outputs = self.getNext(
        self.make_batch_feature(
            filenames=self._filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            reader_num_threads=reader_num_threads,
            parser_num_threads=parser_num_threads))
    self._verify_records(
        batch_size,
        num_epochs=num_epochs,
        interleave_cycle_length=reader_num_threads)
    with self.assertRaises(errors.OutOfRangeError):
      self._next_actual_batch()

  @combinations.generate(
      combinations.times(
          combinations.combine(batch_size=[1, 2], num_epochs=[1, 10]),
          test_base.default_test_combinations()))
  def testDropFinalBatch(self, batch_size, num_epochs):
    # Basic test: read from file 0.
    outputs = self.make_batch_feature(
        filenames=self._filenames[0],
        label_key="label",
        num_epochs=num_epochs,
        batch_size=batch_size,
        drop_final_batch=True)
    for tensor in nest.flatten(outputs):
      if isinstance(tensor, ops.Tensor):  # Guard against SparseTensor.
        self.assertEqual(tensor.shape[0], batch_size)

  @combinations.generate(test_base.default_test_combinations())
  def testIndefiniteRepeatShapeInference(self):
    dataset = self.make_batch_feature(
        filenames=self._filenames[0],
        label_key="label",
        num_epochs=None,
        batch_size=32)
    for shape, clazz in zip(
        nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)),
        nest.flatten(dataset_ops.get_legacy_output_classes(dataset))):
      if issubclass(clazz, ops.Tensor):
        self.assertEqual(32, shape[0])

  @combinations.generate(test_base.default_test_combinations())
  def testOldStyleReader(self):
    with self.assertRaisesRegex(
        TypeError, r"The `reader` argument must return a `Dataset` object. "
        r"`tf.ReaderBase` subclasses are not supported."):
      _ = readers.make_batched_features_dataset(
          file_pattern=self._filenames[0], batch_size=32,
          features={
              "file": parsing_ops.FixedLenFeature([], dtypes.int64),
              "record": parsing_ops.FixedLenFeature([], dtypes.int64),
              "keywords": parsing_ops.VarLenFeature(dtypes.string),
              "label": parsing_ops.FixedLenFeature([], dtypes.string),
          },
          reader=io_ops.TFRecordReader)


if __name__ == "__main__":
  test.main()
