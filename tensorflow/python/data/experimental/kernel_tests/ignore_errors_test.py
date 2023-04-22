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
"""Tests for `tf.data.experimental.ignore_errors()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

_NUMPY_RANDOM_SEED = 42


class IgnoreErrorsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.check_numerics(x, "message")).apply(
            error_ops.ignore_errors()))
    get_next = self.getNext(dataset)

    for x in [1., 2., 3., 5.]:
      self.assertEqual(x, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreError_withLogWarning(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.check_numerics(x, "message")).apply(
                error_ops.ignore_errors(log_warning=True)))
    get_next = self.getNext(dataset)
    for x in [1., 2., 3.]:
      self.assertEqual(x, self.evaluate(get_next()))
    with self.captureWritesToStream(sys.stderr) as logged:
      self.assertEqual(5., self.evaluate(get_next()))
    expected = "Tensor had NaN values"
    self.assertIn((expected), logged.contents())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testParallelMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.check_numerics(x, "message"),
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    get_next = self.getNext(dataset)

    for x in [1., 2., 3., 5.]:
      self.assertEqual(x, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testReadFileIgnoreError(self):

    def write_string_to_file(value, filename):
      with open(filename, "w") as f:
        f.write(value)

    filenames = [
        os.path.join(self.get_temp_dir(), "file_%d.txt" % i) for i in range(5)
    ]
    for filename in filenames:
      write_string_to_file(filename, filename)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(filenames).map(
            io_ops.read_file,
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    get_next = self.getNext(dataset)

    # All of the files are present.
    for filename in filenames:
      self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Delete one of the files.
    os.remove(filenames[0])

    # Attempting to read filenames[0] will fail, but ignore_errors()
    # will catch the error.
    get_next = self.getNext(dataset)
    for filename in filenames[1:]:
      self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDatasetIgnoreError(self):
    filenames = []
    for i in range(5):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for _ in range(10):
        writer.write(b"record")
      writer.close()
      # Append corrupted data
      with open(fn, "a") as f:
        f.write("corrupted data")

    dataset = readers.TFRecordDataset(filenames).apply(
        error_ops.ignore_errors())
    get_next = self.getNext(dataset)

    # All of the files are present.
    for _ in filenames:
      for _ in range(10):
        self.assertEqual(b"record", self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testZipIgnoreError(self):
    a = dataset_ops.Dataset.from_tensor_slices([1., 2., 0., 4.])
    b = a.map(lambda x: array_ops.check_numerics(1. / x, "error"))

    dataset = dataset_ops.Dataset.zip((b, a)).apply(error_ops.ignore_errors())
    get_next = self.getNext(dataset)

    for x in [1., 2., 4.]:
      self.assertEqual((1. / x, x), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    ds = dataset_ops.Dataset.range(10).apply(error_ops.ignore_errors())
    self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.UNKNOWN)


class IgnoreErrorsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                 parameterized.TestCase):

  def _build_ds(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    dataset = dataset.apply(error_ops.ignore_errors())
    options = dataset_ops.Options()
    options.experimental_external_state_policy = (
        distribute_options.ExternalStatePolicy.IGNORE)
    return dataset.with_options(options)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    verify_fn(self, self._build_ds, num_outputs=4)


if __name__ == "__main__":
  test.main()
