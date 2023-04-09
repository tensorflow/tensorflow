# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `MakeDeterministic` optimization."""
import os
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as reader_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MakeDeterministicTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _set_seed(self):
    # Set the seed, since in graph mode some non-random dataset ops call
    # tf.compat.v1.get_seed to copy the seed to a Defun. Calling get_seed raises
    # an error with determinism if no seed is set.
    # TODO(reedwm): Ensure such dataset ops do not raise an error when no seed
    # is set.
    random_seed.set_random_seed(1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_legacy_interleave=[False, True])))
  def test_stateful_ops_interleave(self, use_function, use_legacy_interleave):
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        v.assign_add(1.)
        return (x, v.read_value())

      def interleave_fn(x):
        del x
        return dataset_ops.Dataset.range(2).map(map_fn)

      if use_function:
        map_fn = def_function.function(map_fn)
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_interleave:
        dataset = dataset.apply(
            interleave_ops.parallel_interleave(interleave_fn, cycle_length=5))
      else:
        dataset = dataset.interleave(
            interleave_fn, cycle_length=5, num_parallel_calls=3)
      self.evaluate(variables.global_variables_initializer())
      expected_output = list(zip([0] * 5 + [1] * 5, range(1, 11)))
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True])))
  def test_stateful_ops_map(self, use_function):
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        v.assign_add(1.)
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = list(zip(range(0, 5), range(1, 6)))
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def test_stateful_ops_map_with_random_ops(self):
    with test_util.deterministic_ops():

      def map_fn(x):
        return x + random_ops.random_uniform(
            (), 0, 2, dtype=dtypes.int64, seed=1)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.apply(testing.assert_next(["Map", "ParallelMap"]))
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      get_next = self.getNext(dataset, requires_initialization=True)
      for i in range(5):
        self.assertIn(self.evaluate(get_next()), [i, i + 1])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(use_function=[False, True])))
  def test_stateful_ops_map_ignore_input(self, use_function):
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        del x
        v.assign_add(1.)
        return math_ops.constant_op.constant(1.)

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [1.] * 5
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(use_function=[False, True])))
  def test_stateful_ops_batch(self, use_function):
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn)
      dataset = dataset.apply(testing.assert_next(["Batch"]))
      dataset = dataset.batch(2, num_parallel_calls=2)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [
          (np.array([0, 1]), np.array([0, 0])),
          (np.array([2, 3]), np.array([0, 0])),
          (np.array([4]), np.array([0])),
      ]
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True],
              use_legacy_map_and_batch=[False, True])))
  def test_stateful_ops_map_and_batch(self, use_function,
                                      use_legacy_map_and_batch):
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        v.assign_add(1.)
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_map_and_batch:
        dataset = dataset.apply(batching.map_and_batch(map_fn, 2,
                                                       num_parallel_calls=5))
      else:
        dataset = dataset.map(map_fn, num_parallel_calls=5)
        dataset = dataset.batch(2)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [
          (np.array([0, 1]), np.array([1, 2])),
          (np.array([2, 3]), np.array([3, 4])),
          (np.array([4]), np.array([5])),
      ]
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_legacy_interleave=[False, True])))
  def test_no_stateful_ops_interleave(self, use_function,
                                      use_legacy_interleave):
    self._set_seed()
    with test_util.deterministic_ops():

      def interleave_fn(x):
        del x
        return dataset_ops.Dataset.range(2)

      if use_function:
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_interleave:
        dataset = dataset.apply(
            testing.assert_next(["LegacyParallelInterleaveV2"]))
        dataset = dataset.apply(
            interleave_ops.parallel_interleave(interleave_fn, cycle_length=5))
      else:
        dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
        dataset = dataset.interleave(
            interleave_fn, cycle_length=5, num_parallel_calls=3)
      self.evaluate(variables.global_variables_initializer())
      self.assertDatasetProduces(dataset, expected_output=[0] * 5 + [1] * 5)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(use_function=[False, True])))
  def test_no_stateful_ops_map(self, use_function):
    self._set_seed()
    with test_util.deterministic_ops():
      def map_fn(x):
        return x + 1

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.apply(testing.assert_next(["ParallelMap"]))
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = range(1, 6)
      self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_control_flow=[False, True])))
  def test_text_line_dataset(self, use_function, use_control_flow):
    self._set_seed()
    with test_util.deterministic_ops():

      def write_nums_to_file(filename, numbers):
        path = os.path.join(self.get_temp_dir(), filename)
        with open(path, "w") as f:
          f.write("\n".join(str(n) for n in numbers))
        return path

      f1 = write_nums_to_file("f1", (1, 2, 3))
      f2 = write_nums_to_file("f2", (4, 5, 6))
      f3 = write_nums_to_file("f3", (7, 8, 9))

      if use_control_flow:
        def interleave_fn(filename):
          # Test function that uses control flow. The True branch is never taken
          concat = string_ops.string_join([filename, "abc"])
          return cond.cond(
              math_ops.equal(filename, "abc"),
              lambda: reader_ops.TextLineDataset(concat),
              lambda: reader_ops.TextLineDataset(filename))
      else:
        def interleave_fn(filename):
          return reader_ops.TextLineDataset(filename)

      if use_function:
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.from_tensor_slices([f1, f2, f3])
      dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
      dataset = dataset.interleave(
          interleave_fn, cycle_length=3, num_parallel_calls=3)

      self.assertDatasetProduces(
          dataset,
          expected_output=["1", "4", "7", "2", "5", "8", "3", "6", "9"])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def test_deterministic_attribute(self, local_determinism, global_determinism):
    self._set_seed()
    with test_util.deterministic_ops():

      def sleep(x):
        time.sleep(0.1)
        return x

      def map_function(x):
        if math_ops.equal(x, 0):
          return script_ops.py_func(sleep, [x], x.dtype, stateful=False)
        else:
          return x

      dataset = dataset_ops.Dataset.range(100)
      dataset = dataset.map(
          map_function, num_parallel_calls=2, deterministic=local_determinism)
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)

      self.assertDatasetProduces(dataset, expected_output=range(100))

  @combinations.generate(test_base.default_test_combinations())
  def test_rewrite_prefetch(self):
    with test_util.deterministic_ops():
      v = variables.Variable(-1, dtype=dtypes.int64)

      def map_fn(x):
        v.assign(x)
        return x

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn)
      dataset = dataset.prefetch(5)
      self.evaluate(variables.global_variables_initializer())
      get_next = self.getNext(dataset, requires_initialization=True)
      self.assertEqual(self.evaluate(v), -1)
      self.assertEqual(self.evaluate(get_next()), 0)
      time.sleep(0.01)
      self.assertEqual(self.evaluate(v), 0)
      self.assertEqual(self.evaluate(get_next()), 1)
      time.sleep(0.01)
      self.assertEqual(self.evaluate(v), 1)

  @combinations.generate(test_base.default_test_combinations())
  def test_no_determinism(self):
    config.disable_op_determinism()
    v = variables.Variable(0.)

    def interleave_fn(x):
      del x
      v.assign(1.)
      return dataset_ops.Dataset.range(2)

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
    dataset = dataset.interleave(
        interleave_fn, cycle_length=5, num_parallel_calls=3)
    self.evaluate(variables.global_variables_initializer())
    expected_output = [0] * 5 + [1] * 5
    self.assertDatasetProduces(
        dataset, expected_output=expected_output, requires_initialization=True)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
