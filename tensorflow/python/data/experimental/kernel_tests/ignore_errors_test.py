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

import numpy as np

from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

_NUMPY_RANDOM_SEED = 42


class IgnoreErrorsTest(test_base.DatasetTestBase):

  @test_util.run_deprecated_v1
  def testMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.check_numerics(x, "message")).apply(
            error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)

  @test_util.run_deprecated_v1
  def testParallelMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.check_numerics(x, "message"),
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)

  @test_util.run_deprecated_v1
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
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # All of the files are present.
      self.evaluate(init_op)
      for filename in filenames:
        self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)

      # Delete one of the files.
      os.remove(filenames[0])

      # Attempting to read filenames[0] will fail, but ignore_errors()
      # will catch the error.
      self.evaluate(init_op)
      for filename in filenames[1:]:
        self.assertEqual(compat.as_bytes(filename), self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)


if __name__ == "__main__":
  test.main()
