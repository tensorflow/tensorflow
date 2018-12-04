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

import numpy as np

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class MakeBatchedFeaturesDatasetTest(
    reader_dataset_ops_test_base.MakeBatchedFeaturesDatasetTestBase):

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default() as g:
          with self.session(graph=g) as sess:
            # Basic test: read from file 0.
            self.outputs = dataset_ops.make_one_shot_iterator(
                self.make_batch_feature(
                    filenames=self.test_filenames[0],
                    label_key="label",
                    num_epochs=num_epochs,
                    batch_size=batch_size)).get_next()
            self.verify_records(
                sess,
                batch_size,
                0,
                num_epochs=num_epochs,
                label_key_provided=True)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess, label_key_provided=True)

        with ops.Graph().as_default() as g:
          with self.session(graph=g) as sess:
            # Basic test: read from file 1.
            self.outputs = dataset_ops.make_one_shot_iterator(
                self.make_batch_feature(
                    filenames=self.test_filenames[1],
                    label_key="label",
                    num_epochs=num_epochs,
                    batch_size=batch_size)).get_next()
            self.verify_records(
                sess,
                batch_size,
                1,
                num_epochs=num_epochs,
                label_key_provided=True)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess, label_key_provided=True)

        with ops.Graph().as_default() as g:
          with self.session(graph=g) as sess:
            # Basic test: read from both files.
            self.outputs = dataset_ops.make_one_shot_iterator(
                self.make_batch_feature(
                    filenames=self.test_filenames,
                    label_key="label",
                    num_epochs=num_epochs,
                    batch_size=batch_size)).get_next()
            self.verify_records(
                sess,
                batch_size,
                num_epochs=num_epochs,
                label_key_provided=True)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess, label_key_provided=True)

        with ops.Graph().as_default() as g:
          with self.session(graph=g) as sess:
            # Basic test: read from both files.
            self.outputs = dataset_ops.make_one_shot_iterator(
                self.make_batch_feature(
                    filenames=self.test_filenames,
                    num_epochs=num_epochs,
                    batch_size=batch_size)).get_next()
            self.verify_records(sess, batch_size, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

  @test_util.run_deprecated_v1
  def testReadWithEquivalentDataset(self):
    features = {
        "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        "record": parsing_ops.FixedLenFeature([], dtypes.int64),
    }
    dataset = (
        core_readers.TFRecordDataset(self.test_filenames)
        .map(lambda x: parsing_ops.parse_single_example(x, features))
        .repeat(10).batch(2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for file_batch, _, _, _, record_batch, _ in self._next_expected_batch(
          range(self._num_files), 2, 10):
        actual_batch = self.evaluate(next_element)
        self.assertAllEqual(file_batch, actual_batch["file"])
        self.assertAllEqual(record_batch, actual_batch["record"])
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testReadWithFusedShuffleRepeatDataset(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    for batch_size in [1, 2]:
      # Test that shuffling with same seed produces the same result.
      with ops.Graph().as_default() as g:
        with self.session(graph=g) as sess:
          outputs1 = dataset_ops.make_one_shot_iterator(self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)).get_next()
          outputs2 = dataset_ops.make_one_shot_iterator(self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)).get_next()
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      # Test that shuffling with different seeds produces a different order.
      with ops.Graph().as_default() as g:
        with self.session(graph=g) as sess:
          outputs1 = dataset_ops.make_one_shot_iterator(self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5)).get_next()
          outputs2 = dataset_ops.make_one_shot_iterator(self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=15)).get_next()
          all_equal = True
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)

  def testParallelReadersAndParsers(self):
    num_epochs = 5
    for batch_size in [1, 2]:
      for reader_num_threads in [2, 4]:
        for parser_num_threads in [2, 4]:
          with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
              self.outputs = dataset_ops.make_one_shot_iterator(
                  self.make_batch_feature(
                      filenames=self.test_filenames,
                      label_key="label",
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      reader_num_threads=reader_num_threads,
                      parser_num_threads=parser_num_threads)).get_next()
              self.verify_records(
                  sess,
                  batch_size,
                  num_epochs=num_epochs,
                  label_key_provided=True,
                  interleave_cycle_length=reader_num_threads)
              with self.assertRaises(errors.OutOfRangeError):
                self._next_actual_batch(sess, label_key_provided=True)

          with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
              self.outputs = dataset_ops.make_one_shot_iterator(
                  self.make_batch_feature(
                      filenames=self.test_filenames,
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      reader_num_threads=reader_num_threads,
                      parser_num_threads=parser_num_threads)).get_next()
              self.verify_records(
                  sess,
                  batch_size,
                  num_epochs=num_epochs,
                  interleave_cycle_length=reader_num_threads)
              with self.assertRaises(errors.OutOfRangeError):
                self._next_actual_batch(sess)

  def testDropFinalBatch(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default():
          # Basic test: read from file 0.
          outputs = dataset_ops.make_one_shot_iterator(self.make_batch_feature(
              filenames=self.test_filenames[0],
              label_key="label",
              num_epochs=num_epochs,
              batch_size=batch_size,
              drop_final_batch=True)).get_next()
          for tensor in nest.flatten(outputs):
            if isinstance(tensor, ops.Tensor):  # Guard against SparseTensor.
              self.assertEqual(tensor.shape[0], batch_size)

  def testIndefiniteRepeatShapeInference(self):
    dataset = self.make_batch_feature(
        filenames=self.test_filenames[0],
        label_key="label",
        num_epochs=None,
        batch_size=32)
    for shape, clazz in zip(nest.flatten(dataset.output_shapes),
                            nest.flatten(dataset.output_classes)):
      if issubclass(clazz, ops.Tensor):
        self.assertEqual(32, shape[0])

  def testOldStyleReader(self):
    with self.assertRaisesRegexp(
        TypeError, r"The `reader` argument must return a `Dataset` object. "
        r"`tf.ReaderBase` subclasses are not supported."):
      _ = readers.make_batched_features_dataset(
          file_pattern=self.test_filenames[0], batch_size=32,
          features={
              "file": parsing_ops.FixedLenFeature([], dtypes.int64),
              "record": parsing_ops.FixedLenFeature([], dtypes.int64),
              "keywords": parsing_ops.VarLenFeature(dtypes.string),
              "label": parsing_ops.FixedLenFeature([], dtypes.string),
          },
          reader=io_ops.TFRecordReader)


if __name__ == "__main__":
  test.main()
