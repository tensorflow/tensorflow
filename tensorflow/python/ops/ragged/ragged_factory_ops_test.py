# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that ragged tensors work with GPU, such as placement of int and string.

Test using ragged tensors with map_fn and distributed dataset. Since GPU does
not support strings, ragged tensors containing string should always be placed
on CPU.
"""

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import map_fn
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.string_ops import string_to_hash_bucket
from tensorflow.python.platform import test


def ragged_int64():
  return ragged_factory_ops.constant(
      [
          [3, 1, 4, 1],
          [],
          [5, 9, 2],
          [6],
          [],
          [3, 1, 4, 1],
          [3, 1],
          [2, 1, 4, 1],
      ],
      dtype=dtypes.int64,
  )


def ragged_str():
  return ragged_factory_ops.constant([
      ['3', '1', '4', '1'],
      [],
      ['5', '9', '2'],
      ['6'],
      [],
      ['3', '1', '4', '1'],
      ['3', '1'],
      ['2', '1', '4', '1'],
  ])


def dense_str():
  return constant_op.constant([
      ['3', '1', '4', '1'],
      ['', '', '', ''],
      ['5', '9', '2', ''],
      ['6', '', '', ''],
      ['', '', '', ''],
      ['3', '1', '4', '1'],
      ['3', '1', '', ''],
      ['2', '1', '4', '1'],
  ])


class RaggedFactoryOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  @parameterized.parameters(
      (ragged_int64,),
      (ragged_str,),
  )
  def testRaggedWithMapFn(self, ragged_factory):

    @def_function.function
    def map_fn_producer(inputs):
      return map_fn.map_fn_v2(lambda x: x, inputs)

    t = ragged_factory()
    result = self.evaluate(map_fn_producer(t))
    self.assertAllEqual(t.values, result.values)

  @parameterized.parameters(
      (ragged_int64,),
      (ragged_str,),
  )
  def testRaggedWithMultiDeviceIterator(self, ragged_factory):

    @def_function.function
    def dataset_producer(t):
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      it = multi_device_iterator_ops.MultiDeviceIterator(ragged_ds, ['GPU:0'])
      with ops.device_v2('GPU:0'):
        return it.get_next_as_optional()

    t = ragged_factory()
    if t.dtype == dtypes.string:
      self.skipTest('b/194439197: fix ragged tensor of string')
    result = dataset_producer(t)
    self.assertAllEqual(
        self.evaluate(t[0]), self.evaluate(result[0].get_value()[0]))

  @parameterized.parameters(
      (ragged_int64,),
      (ragged_str,),
  )
  def testRaggedWithDistributedDataset(self, ragged_factory):

    @def_function.function
    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)
      ds = iter(dist_dataset)
      return strategy.experimental_local_results(next(ds))[0]

    t = ragged_factory()
    if t.dtype == dtypes.string:
      self.skipTest('b/194439197: fix ragged tensor of string')

    result = distributed_dataset_producer(t)
    self.assertAllEqual(self.evaluate(t[0]), self.evaluate(result[0]))

  @parameterized.parameters(
      (dense_str,),
      # (ragged_str,),  # TODO(b/194439197) fix ragged tensor of string
  )
  def testIntStringWithDistributedDataset(self, string_factory):

    @def_function.function
    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)
      ds = iter(dist_dataset)
      return strategy.experimental_local_results(next(ds))[0]

    ds_dict = {'int': ragged_int64(), 'str': string_factory()}
    result = distributed_dataset_producer(ds_dict)
    self.assertAllEqual(
        self.evaluate(ds_dict['int'][0]), self.evaluate(result['int'][0]))
    self.assertAllEqual(
        self.evaluate(ds_dict['str'][0]), self.evaluate(result['str'][0]))

  @parameterized.parameters(
      (dense_str,),
      # (ragged_str,),  # TODO(b/194439197) fix ragged tensor of string
  )
  @test_util.run_v2_only
  def testOpsWithDistributedDataset(self, string_factory):

    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)

      @def_function.function
      def replica_fn(elem):
        # Example of typical preprocessing of string to numeric feature
        hashed = string_to_hash_bucket(elem['str'], 10)
        return 1000 * hashed

      result = []
      for x in dist_dataset:
        result.append(strategy.run(replica_fn, args=(x,)))
      return result

    ds_dict = {'str': string_factory()}
    result = distributed_dataset_producer(ds_dict)
    expect_length = [len(i) for i in ds_dict['str']]
    self.assertAllEqual([[5000, 3000, 5000, 3000][:expect_length[0]]],
                        self.evaluate(result[0].values[0]))
    self.assertAllEqual([[9000, 9000, 9000, 9000][:expect_length[1]]],
                        self.evaluate(result[0].values[1]))
    self.assertAllEqual([[0, 3000, 2000, 9000][:expect_length[2]]],
                        self.evaluate(result[1].values[0]))
    self.assertAllEqual([[2000, 9000, 9000, 9000][:expect_length[3]]],
                        self.evaluate(result[1].values[1]))
    self.assertAllEqual([[9000, 9000, 9000, 9000][:expect_length[4]]],
                        self.evaluate(result[2].values[0]))
    self.assertAllEqual([[5000, 3000, 5000, 3000][:expect_length[5]]],
                        self.evaluate(result[2].values[1]))
    self.assertAllEqual([[5000, 3000, 9000, 9000][:expect_length[6]]],
                        self.evaluate(result[3].values[0]))
    self.assertAllEqual([[2000, 3000, 5000, 3000][:expect_length[7]]],
                        self.evaluate(result[3].values[1]))

  @parameterized.parameters(
      (dense_str,),
      # (ragged_str,),  # TODO(b/194439197) fix ragged tensor of string
  )
  @test_util.run_v2_only
  def testIntStringOpsWithDistributedDataset(self, string_factory):

    ri = ragged_int64()
    # To ease testing both dense and ragged strings, pass in the ragged sizes
    # so the dense strings can be sliced to match.
    element_sizes = [len(i) for i in ri]
    ds_dict = {'int': ri, 'str': string_factory(), 'size': element_sizes}

    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)

      @def_function.function
      def replica_fn(elem):
        # Example of typical preprocessing of string to numeric feature
        hashed = string_to_hash_bucket(elem['str'], 10)
        # For dense string case, slice it to size of ragged int
        hashed_sliced = hashed[:, :elem['size'][0]]
        # Computation with both feature from string and numeric dataset output
        return elem['int'] * 10 + hashed_sliced

      result = []
      for x in dist_dataset:
        result.append(strategy.run(replica_fn, args=(x,)))
      return result

    result = distributed_dataset_producer(ds_dict)
    self.assertAllEqual([[35, 13, 45, 13]], self.evaluate(result[0].values[0]))
    self.assertAllEqual([[]], self.evaluate(result[0].values[1]))
    self.assertAllEqual([[50, 93, 22]], self.evaluate(result[1].values[0]))
    self.assertAllEqual([[62]], self.evaluate(result[1].values[1]))
    self.assertAllEqual([[]], self.evaluate(result[2].values[0]))
    self.assertAllEqual([[35, 13, 45, 13]], self.evaluate(result[2].values[1]))
    self.assertAllEqual([[35, 13]], self.evaluate(result[3].values[0]))
    self.assertAllEqual([[22, 13, 45, 13]], self.evaluate(result[3].values[1]))


if __name__ == '__main__':
  test.main()
