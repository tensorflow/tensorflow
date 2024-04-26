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
"""Tests for `tf.data.Dataset.flat_map()`."""
import random
from typing import Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class FlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):

  # pylint: disable=g-long-lambda
  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapDataset(self):
    repeats = [1, 2, 3, 4, 5, 0, 1]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensors([x]).repeat(x)
    )
    expected_output = []
    for i in repeats:
      expected_output.extend([[i]] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
            lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y)
        )
    )
    expected_output = []
    for row in repeats:
      for i in row:
        expected_output.extend([i] * i)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.graph_only_combinations())
  def testSharedResourceNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.from_tensor_slices(components).flat_map(
            lambda x: dataset_ops.Dataset.from_tensor_slices(x).flat_map(
                lambda y: dataset_ops.Dataset.from_tensors(y).repeat(y)
            )
        ),
        shared_name="shared_flat_map_iterator",
    )
    init_op = iterator.initializer
    get_next = iterator.get_next()

    # Create two concurrent sessions that share the same iterator
    # resource on the same server, and verify that a random
    # interleaving of `Session.run(get_next)` calls on the two
    # sessions yields the expected result.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess1:
      with session.Session(server.target) as sess2:
        for _ in range(3):
          sess = random.choice([sess1, sess2])
          sess.run(init_op)
          for row in repeats:
            for i in row:
              for _ in range(i):
                sess = random.choice([sess1, sess2])
                self.assertEqual(i, sess.run(get_next))

        with self.assertRaises(errors.OutOfRangeError):
          sess = random.choice([sess1, sess2])
          sess.run(get_next)

  @combinations.generate(test_base.default_test_combinations())
  def testMapDict(self):
    dataset = (
        dataset_ops.Dataset.range(10)
        .map(lambda x: {"foo": x * 2, "bar": x**2})
        .flat_map(
            lambda d: dataset_ops.Dataset.from_tensors(d["foo"]).repeat(
                d["bar"]
            )
        )
    )
    get_next = self.getNext(dataset)
    for i in range(10):
      for _ in range(i**2):
        self.assertEqual(i * 2, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2]
      )

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values)
      )

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      for j in range(2):
        expected_output.append([i, 0] if j % 2 == 0 else [0, -i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArray(self):
    def _map_fn(i):
      i = math_ops.cast(i, dtypes.int32)
      return tensor_array_ops.TensorArray(
          dtype=dtypes.int32, element_shape=(), size=i
      ).unstack(math_ops.range(i))

    def _flat_map_fn(x):
      self.assertIsInstance(x, tensor_array_ops.TensorArray)
      return dataset_ops.Dataset.from_tensor_slices(x.stack())

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    expected_output = []
    for i in range(10):
      for j in range(i):
        expected_output.append(j)

    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRagged(self):
    def _map_fn(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1], [-1]])

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          ragged_conversion_ops.to_tensor(x)
      )

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)
    expected_output = []
    for i in range(10):
      expected_output.append([i])
      expected_output.append([-i])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    def fn(x):
      return dataset_ops.Dataset.from_tensors(x)

    dataset = dataset_ops.Dataset.from_tensors(42).flat_map(fn, name="flat_map")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), 9)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteCardinality(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(lambda _: dataset_ops.Dataset.range(1).repeat())
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinality(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(
        lambda _: dataset_ops.Dataset.range(10).filter(lambda x: x % 2 == 1))
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.UNKNOWN)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyCardinality(self):
    dataset = dataset_ops.Dataset.range(0)
    dataset = dataset.flat_map(dataset_ops.Dataset.range)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_MODERATE")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testCardinalityLowEffort(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    options = dataset_options_pb2.CardinalityOptions(
        compute_level="CARDINALITY_COMPUTE_LOW")
    cardinality = dataset_ops.gen_dataset_ops.dataset_cardinality(
        dataset._variant_tensor, options.SerializeToString())
    self.assertEqual(self.evaluate(cardinality), dataset_ops.UNKNOWN)

  @combinations.generate(test_base.default_test_combinations())
  def testMapFuncFailWithErrorContext(self):

    def fn(x):
      return dataset_ops.Dataset.from_tensors(x // 0)

    dataset = dataset_ops.Dataset.from_tensors(42).flat_map(fn, name="flat_map")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())

  @combinations.generate(test_base.v2_eager_only_combinations())
  def testSymbolicCheckpointSize(self):
    examples_per_flat_map = 100
    example_len = 10_000

    def flat_map_fn(_):
      data = []
      for _ in range(examples_per_flat_map):
        data.append(
            stateless_random_ops.stateless_random_uniform(
                [example_len], seed=(42, 42)
            )
        )
      return dataset_ops.Dataset.from_tensor_slices(data)

    ds = dataset_ops.Dataset.range(10)
    # Inputs to flat_map are >1MB
    ds = ds.map(
        lambda x: stateless_random_ops.stateless_random_uniform(
            [1_000_000], seed=(42, 42)
        )
    )
    ds = ds.flat_map(flat_map_fn)

    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = True
    ds = ds.with_options(options)

    it = ds.as_numpy_iterator()
    for _ in range(30):
      next(it)

    ckpt = it._save()
    # Make sure the checkpoint is smaller than the element sizes, i.e. no
    # elements are being stored in the checkpoint.
    self.assertLess(len(ckpt.numpy()), 10_000)


class FlatMapCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
      )
  )
  def test(self, verify_fn, symbolic_checkpoint):
    # Complicated way of saying range(start, start+25).
    def build_ds(start):
      def map_fn(x):
        return dataset_ops.Dataset.range(x, x + 5)

      dataset = dataset_ops.Dataset.range(start, start + 5 * 5, 5)
      dataset = dataset.flat_map(map_fn)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, lambda: build_ds(0), num_outputs=25)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testNested(self, verify_fn):
    def build_ds():
      inner_ds = dataset_ops.Dataset.from_tensor_slices(range(42))
      ds = dataset_ops.Dataset.from_tensors(inner_ds)
      return ds.flat_map(lambda x: x)

    verify_fn(self, build_ds, num_outputs=42)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testMapThenFlatMap(self, verify_fn):
    def build_ds():
      def flat_map_fn(_):
        def map_fn(y):
          return 10 * math_ops.cast(y, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    verify_fn(self, build_ds, num_outputs=500)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testCaptureDefunInMapFn(self, verify_fn):
    def build_ds():
      def map_fn(x):
        @function.Defun(dtypes.int64)
        def defun_fn(x):
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.from_tensor_slices([defun_fn(x)])

      return dataset_ops.Dataset.range(100).flat_map(map_fn)

    verify_fn(self, build_ds, num_outputs=100)

  @combinations.generate(test_base.default_test_combinations())
  def testDisallowVariableCapture(self):
    def build_ds():
      test_var = variable_scope.get_variable(
          name="test_var", shape=(), use_resource=True
      )
      return dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensor_slices([test_var])
      )

    self.verify_error_on_save(build_ds, 5, errors.FailedPreconditionError)

  @combinations.generate(test_base.default_test_combinations())
  def testDisallowCapturingStatefulOps(self):
    def build_ds():
      def flat_map_fn(_):
        def map_fn(x):
          return random_ops.random_uniform(
              (), 0, 10, dtype=dtypes.int32
          ) * math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    self.verify_error_on_save(build_ds, 500, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testSparse(self, verify_fn):
    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2]
      )

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values)
      )

    def _build_ds():
      return dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    verify_fn(self, _build_ds, num_outputs=20)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[True],
                               num_skips=[3, 4]),
      )
  )
  def testWithSkip(self, verify_fn, symbolic_checkpoint, num_skips):
    """Test `.flat_map().skip()` checkpointing behavior.

    `SkipInternal` and `GetNextInternal` are separate functions
    but with slighly different implementations.
    Therefore, we should test this op's behavior when used with `.skip()`.

    Args:
      verify_fn: Verify the correctness of this dataset's checkpointing.
      symbolic_checkpoint: Whether symbolic checkpointing is turned on.
      num_skips: `.skip(num_skips)`
    """

    def build_dataset():
      def my_map(x):
        if x == 0:
          return dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3])
        elif x == 1:
          return dataset_ops.Dataset.from_tensor_slices([4, 5, 6, 7])
        else:
          return dataset_ops.Dataset.from_tensor_slices([8, 9, 10, 11])

      indices = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
      dataset = indices.flat_map(my_map)
      # Skip some elements
      dataset = dataset.skip(num_skips)

      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, build_dataset, num_outputs=3 * 4 - num_skips)


class FlatMapGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              repetitions=[1, 2],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def test(
      self,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(1, 10)) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))


if __name__ == "__main__":
  test.main()
