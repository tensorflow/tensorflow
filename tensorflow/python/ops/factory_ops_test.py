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
"""Tests that sparse tensors work with GPU, such as placement of int and string.

Test using sparse tensors with distributed dataset. Since GPU does
not support strings, sparse tensors containing string should always be placed
on CPU.
"""

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def sparse_int64():
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.int64),
      dense_shape=[8, 4])


def sparse_str():
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant(['1', '2', '3', '4', '5', '6', '7', '8']),
      dense_shape=[8, 4])


class FactoryOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters(
      (sparse_int64,),
      (sparse_str,),
  )
  @test_util.run_gpu_only
  def testSparseWithDistributedDataset(self, sparse_factory):

    @def_function.function
    def distributed_dataset_producer(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      sparse_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(sparse_ds)
      ds = iter(dist_dataset)
      result = strategy.experimental_local_results(next(ds))[0]
      # Reach the end of the iterator
      for ignore in ds:  # pylint: disable=unused-variable
        pass
      return result

    t = sparse_factory()

    result = distributed_dataset_producer(t)
    self.assertAllEqual(
        self.evaluate(sparse_ops.sparse_tensor_to_dense(t)[0]),
        self.evaluate(sparse_ops.sparse_tensor_to_dense(result)[0]))


if __name__ == '__main__':
  test.main()
