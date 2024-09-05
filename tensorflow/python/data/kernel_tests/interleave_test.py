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
"""Tests for `tf.data.Dataset.interleave()`."""
import multiprocessing
import os
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


def _interleave(lists, cycle_length, block_length, num_parallel_calls=None):
  """Reference implementation of interleave used for testing.

  Args:
    lists: a list of lists to interleave
    cycle_length: the length of the interleave cycle
    block_length: the length of the interleave block
    num_parallel_calls: the number of parallel calls

  Yields:
    Elements of `lists` interleaved in the order determined by `cycle_length`
    and `block_length`.
  """
  num_open = 0

  # `all_iterators` acts as a queue of iterators over each element of `lists`.
  all_iterators = [iter(l) for l in lists]

  # `open_iterators` are the iterators whose elements are currently being
  # interleaved.
  open_iterators = []
  if cycle_length is None:
    # The logic here needs to match interleave C++ kernels.
    cpu_count = multiprocessing.cpu_count()
    if hasattr(os, "sched_getaffinity"):
      try:
        cpu_count = len(os.sched_getaffinity(0))
      except NotImplementedError:
        pass
    if num_parallel_calls is None:
      cycle_length = cpu_count
    elif num_parallel_calls == dataset_ops.AUTOTUNE:
      cycle_length = (cpu_count + 2) // 3
    else:
      cycle_length = min(num_parallel_calls, cpu_count)

  for i in range(cycle_length):
    if all_iterators:
      open_iterators.append(all_iterators.pop(0))
      num_open += 1
    else:
      open_iterators.append(None)

  while num_open or all_iterators:
    for i in range(cycle_length):
      if open_iterators[i] is None:
        if all_iterators:
          open_iterators[i] = all_iterators.pop(0)
          num_open += 1
        else:
          continue
      for _ in range(block_length):
        try:
          yield next(open_iterators[i])
        except StopIteration:
          open_iterators[i] = None
          num_open -= 1
          break


def _repeat(values, count):
  """Produces a list of lists suitable for testing interleave.

  Args:
    values: for each element `x` the result contains `[x] * x`
    count: determines how many times to repeat `[x] * x` in the result

  Returns:
    A list of lists of values suitable for testing interleave.
  """
  return [[value] * value for value in np.tile(values, count)]


class InterleaveTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[[4, 5, 6]],
              cycle_length=1,
              block_length=1,
              expected_elements=[[
                  4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 5, 5,
                  5, 5, 5, 6, 6, 6, 6, 6, 6
              ]]) + combinations.combine(
                  input_values=[[4, 5, 6]],
                  cycle_length=2,
                  block_length=1,
                  expected_elements=[[
                      4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6,
                      5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 6
                  ]]) + combinations.combine(
                      input_values=[[4, 5, 6]],
                      cycle_length=2,
                      block_length=3,
                      expected_elements=[[
                          4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 6, 6, 4, 4, 4, 6, 6, 6,
                          4, 5, 5, 5, 6, 6, 6, 5, 5, 6, 6, 6
                      ]]) + combinations.combine(
                          input_values=[[4, 5, 6]],
                          cycle_length=7,
                          block_length=2,
                          expected_elements=[[
                              4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6,
                              6, 4, 4, 5, 5, 6, 6, 5, 6, 6, 5, 6, 6
                          ]]) +
          combinations.combine(
              input_values=[[4, 0, 6]],
              cycle_length=2,
              block_length=1,
              expected_elements=[[
                  4, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6
              ]])))
  def testPythonImplementation(self, input_values, cycle_length, block_length,
                               expected_elements):
    input_lists = _repeat(input_values, 2)

    for expected, produced in zip(
        expected_elements, _interleave(input_lists, cycle_length,
                                       block_length)):
      self.assertEqual(expected, produced)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.int64([4, 5, 6])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.int64([4, 5, 6])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[None, 1, 2]) + combinations.combine(
                      input_values=[np.int64([4, 5, 6])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[None, 1, 3, 5, 7]) +
          combinations.combine(
              input_values=[np.int64([4, 5, 6, 7])],
              cycle_length=None,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.int64([]), np.int64([0, 0, 0])],
                  cycle_length=2,
                  block_length=3,
                  num_parallel_calls=[None]) + combinations.combine(
                      input_values=[np.int64([4, 0, 6])],
                      cycle_length=2,
                      block_length=3,
                      num_parallel_calls=[None, 1, 2])))
  def testInterleaveDataset(self, input_values, cycle_length, block_length,
                            num_parallel_calls):
    count = 2
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        count).interleave(
            lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
            cycle_length, block_length, num_parallel_calls)
    expected_output = [
        element for element in _interleave(
            _repeat(input_values, count), cycle_length, block_length,
            num_parallel_calls)
    ]
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[None, 1, 2]) + combinations.combine(
                      input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[None, 1, 3, 5, 7])))
  def testInterleaveDatasetError(self, input_values, cycle_length, block_length,
                                 num_parallel_calls):
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).map(
        lambda x: array_ops.check_numerics(x, "message")).interleave(
            dataset_ops.Dataset.from_tensors, cycle_length, block_length,
            num_parallel_calls)
    get_next = self.getNext(dataset)

    for value in input_values:
      if np.isnan(value):
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(get_next())
      else:
        self.assertEqual(value, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveSparse(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).interleave(
        _interleave_fn, cycle_length=1)
    get_next = self.getNext(dataset)
    for i in range(10):
      for j in range(2):
        expected = [i, 0] if j % 2 == 0 else [0, -i]
        self.assertAllEqual(expected, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.int64([4, 5, 6])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=1) + combinations.combine(
                  input_values=[np.int64([4, 5, 6])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[1, 2]) + combinations.combine(
                      input_values=[np.int64([4, 5, 6])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[1, 3, 5, 7]) + combinations.combine(
                          input_values=[np.int64([4, 5, 6, 7])],
                          cycle_length=None,
                          block_length=3,
                          num_parallel_calls=1) + combinations.combine(
                              input_values=[np.int64([4, 0, 6])],
                              cycle_length=2,
                              block_length=3,
                              num_parallel_calls=[1, 2])))
  def testSloppyInterleaveDataset(self, input_values, cycle_length,
                                  block_length, num_parallel_calls):
    count = 2
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        count).interleave(
            lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
            cycle_length, block_length, num_parallel_calls)
    options = options_lib.Options()
    options.deterministic = False
    dataset = dataset.with_options(options)
    expected_output = [
        element for element in _interleave(
            _repeat(input_values, count), cycle_length, block_length,
            num_parallel_calls)
    ]
    get_next = self.getNext(dataset)
    actual_output = []
    for _ in range(len(expected_output)):
      actual_output.append(self.evaluate(get_next()))
    self.assertAllEqual(expected_output.sort(), actual_output.sort())

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveMap(self):
    dataset = dataset_ops.Dataset.range(100)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      return dataset.map(lambda x: x + x)

    dataset = dataset.interleave(interleave_fn, cycle_length=5)
    dataset = dataset.interleave(interleave_fn, cycle_length=5)

    self.assertDatasetProduces(dataset, [4 * x for x in range(100)])

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleaveCached(self):
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.cache(os.path.join(self.get_temp_dir(), "cache_dir"))

    def interleave_fn(x):
      return dataset_ops.Dataset.from_tensors(x)

    dataset = dataset.interleave(
        interleave_fn, cycle_length=2, num_parallel_calls=2)
    self.assertDatasetProduces(dataset, list(range(5)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def testDeterminismConfiguration(self, local_determinism, global_determinism):
    expect_determinism = local_determinism or (local_determinism is None and
                                               global_determinism)
    elements = list(range(1000))

    def dataset_fn(delay_ms):

      def interleave_fn(x):
        ds = dataset_ops.Dataset.from_tensors(x)
        if math_ops.equal(x, 0):
          ds = ds.apply(testing.sleep(delay_ms * 1000))
        else:
          ds = ds.apply(testing.sleep(0))
        return ds

      dataset = dataset_ops.Dataset.from_tensor_slices(elements)
      dataset = dataset.interleave(
          interleave_fn,
          cycle_length=10,
          num_parallel_calls=10,
          deterministic=local_determinism)
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)
      return dataset

    self.checkDeterminism(dataset_fn, expect_determinism, elements)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testName(self, num_parallel_calls):

    def fn(x):
      return dataset_ops.Dataset.from_tensors(x)

    dataset = dataset_ops.Dataset.from_tensors(42).interleave(
        fn, num_parallel_calls=num_parallel_calls, name="interleave")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testMapFuncMustReturnDataset(self, num_parallel_calls):

    def map_fn(x):
      return [x]

    with self.assertRaisesRegex(
        TypeError, "The `map_func` argument must return a `Dataset` object."):
      dataset_ops.Dataset.from_tensors(42).interleave(
          map_fn, num_parallel_calls=num_parallel_calls)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testMapFuncFailWithErrorContext(self, num_parallel_calls):

    def fn(x):
      return dataset_ops.Dataset.from_tensors(x // 0)

    dataset = dataset_ops.Dataset.from_tensors(42).interleave(
        fn, num_parallel_calls=num_parallel_calls, name="interleave")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testSymbolicCheckpointSize(self):
    if sys.platform == "darwin":
      self.skipTest(
          "MacOS does not support symbolic checkpointing."
      )  # b/284304023

    dataset = dataset_ops.Dataset.range(10)
    # Each input element to `.interleave` is > 1MB
    dataset = dataset.map(
        # Create a huge input element
        lambda x: stateless_random_ops.stateless_random_uniform(
            [1_000_000], seed=(42, 42)
        )
    )
    dataset = dataset.interleave(
        lambda x: dataset_ops.Dataset.range(200),
        cycle_length=5,
        num_parallel_calls=None,
    )

    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = True
    dataset = dataset.with_options(options)

    it = dataset.as_numpy_iterator()
    for _ in range(5):
      next(it)

    checkpoint = it.save().numpy()
    self.assertLess(
        len(checkpoint),
        5_000,
        f"The checkpoint should be small enough. Got {len(checkpoint)} bytes",
    )


class InterleaveCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              symbolic_checkpoint=[False, True],
              cycle_length=2,
              block_length=[1, 3],
              num_parallel_calls=[None, 1, 2])))
  def test(self, verify_fn, symbolic_checkpoint, cycle_length, block_length,
           num_parallel_calls):

    num_repeats = 2
    input_values = np.array([2, 3], dtype=np.int64)

    def _build_dataset():
      dataset = dataset_ops.Dataset.from_tensor_slices(input_values)
      dataset = dataset.repeat(num_repeats)
      dataset = dataset.interleave(
          lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x), cycle_length,
          block_length, num_parallel_calls)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      dataset = dataset.with_options(options)
      return dataset

    num_outputs = np.sum(input_values) * num_repeats
    verify_fn(self, _build_dataset, num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              skip=[0, 1, 2, 3],
          ),
      )
  )
  def testWithSkip(self, verify_fn, skip):
    def _build_dataset():
      dataset = dataset_ops.Dataset.range(4)

      dataset = dataset.interleave(
          lambda x: dataset_ops.Dataset.from_tensors(x).repeat(3),
          cycle_length=2,
          block_length=1,
          num_parallel_calls=None,
      )
      dataset = dataset.skip(skip)
      return dataset

    num_outputs = 4 * 3 - skip
    verify_fn(self, _build_dataset, num_outputs)

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testDelayedPurgeCheckpointAtTheSameCycleIdx(self):
    """Tests delayed checkpoint purging at the same cycle index works correctly.

    This would crash if we were to use`cycle_index_` as part
    of the prefix:
           [0]                                 [1]
            1(prefix: ::Interleave[0]           2(prefix: ::Interleave[1])
           EOF(delete ::Interleave[0])         EOF(delete ::Interleave[1])
            3(prefix  ::Interleave[2])          4
                                   ^
                                  (should be 2 instead of 0)
           EOF                                 EOF

    If we checkpoint at the point right after 3 is generated and
    restore it, restore would crash because the sub iterator
    for generating 3 is incorrectly deleted due to delayed checkpoint purging.
    """

    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = True
    options.experimental_optimization.inject_prefetch = False
    options.experimental_optimization.apply_default_optimizations = False

    def _build_dataset():
      dataset = dataset_ops.Dataset.range(4)

      dataset = dataset.interleave(
          lambda x: dataset_ops.Dataset.from_tensor_slices([x]),
          cycle_length=2,
          block_length=1,
          num_parallel_calls=None,
      )
      dataset = dataset.with_options(options)
      return dataset

    dataset = _build_dataset().with_options(options)

    it = dataset.as_numpy_iterator()

    for _ in range(3):
      next(it)

    checkpoint = it.save().numpy()

    expected = next(it)

    restored_it = dataset.as_numpy_iterator()
    restored_it.restore(checkpoint)

    actual = next(restored_it)

    self.assertEqual(expected, actual)

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testWithInputThatPurgeCheckpoint(self):
    """Tests underlying `expired_prefixes` are handled correctly.

    Explanation:
        The input for `interleave` looks like (created by `.repeat`):
        [0, |1, |2]
            ^   ^
            |   |
            |   expired_prefixes=["FiniteRepeat[1]"]
            expired_prefixes=["FiniteRepeat[0]"]

        [0]   [1]
         0     1    <--- expired_prefixes=["...FiniteRepeat[0]"]
        EOF   EOF
         2 <----- Tests the previous checkpoint stored at this index
                  should not have an effect on the new checkpoint.

        EOF
    """
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = True
    options.experimental_optimization.inject_prefetch = False
    options.experimental_optimization.apply_default_optimizations = False

    def carefully_designed_map(x):
      if x == 0:
        return dataset_ops.Dataset.from_tensor_slices([0])
      elif x == 1:
        return dataset_ops.Dataset.from_tensor_slices([1])
      else:
        return dataset_ops.Dataset.from_tensor_slices([2])

    def _build_dataset():
      dataset = dataset_ops.Dataset.from_tensor_slices(["does not matter"])

      # Create [0, 1, 2] using repeat+enumerate+map
      dataset = dataset.repeat(3)
      dataset = dataset.enumerate()
      dataset = dataset.map(lambda idx, x: idx)

      dataset = dataset.interleave(
          carefully_designed_map,
          cycle_length=2,
          block_length=1,
          num_parallel_calls=None,
      )
      dataset = dataset.with_options(options)
      return dataset

    dataset = _build_dataset().with_options(options)

    it = dataset.as_numpy_iterator()

    try:
      for _ in range(4):
        next(it)
    except StopIteration:
      pass

    # should not crash
    it.save().numpy()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(num_parallel_calls=[None, 2]),
      )
  )
  def testNested(self, verify_fn, num_parallel_calls):

    def build_ds():

      inner_ds = dataset_ops.Dataset.from_tensor_slices(range(10))
      ds = dataset_ops.Dataset.from_tensors(inner_ds).repeat(10)
      return ds.interleave(
          lambda x: x, cycle_length=5, num_parallel_calls=num_parallel_calls)

    verify_fn(self, build_ds, num_outputs=100)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    def _build_dataset():
      return dataset_ops.Dataset.range(10).map(_map_fn).interleave(
          _interleave_fn, cycle_length=1)

    verify_fn(self, _build_dataset, num_outputs=20)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(num_parallel_calls=[None, 2]),
      )
  )
  def testSymbolicUnimplemented(self, verify_fn, num_parallel_calls):
    if sys.platform == "darwin":
      self.skipTest(
          "MacOS does not support symbolic checkpointing."
      )  # b/284304023

    def fn(x):
      del x
      dataset = dataset_ops.Dataset.range(7)
      dataset = dataset.window(3, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda x: x)
      return dataset

    def build_ds():
      dataset = dataset_ops.Dataset.range(2)
      dataset = dataset.interleave(
          fn,
          cycle_length=3,
          num_parallel_calls=num_parallel_calls,
      )
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = True
      dataset = dataset.with_options(options)
      return dataset

    with self.assertRaisesRegex(
        errors.UnimplementedError,
        "WindowOp does not support symbolic checkpointing.",
    ):
      verify_fn(self, build_ds, num_outputs=30)


if __name__ == "__main__":
  test.main()
