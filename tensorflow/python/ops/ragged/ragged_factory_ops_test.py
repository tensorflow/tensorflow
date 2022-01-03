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
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import map_fn
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _ragged_int():
  return ragged_factory_ops.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], [],
                                      [3, 1, 4, 1], [3, 1], [2, 1, 4, 1]],
                                     dtype=dtypes.int64)


def _ragged_str():
  return ragged_factory_ops.constant([['3', '1', '4', '1'], [], ['5', '9', '2'],
                                      ['6'], [], ['3', '1', '4', '1'],
                                      ['3', '1'], ['2', '1', '4', '1']])


class RaggedFactoryOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='Int', factory_func=_ragged_int),
      dict(testcase_name='String', factory_func=_ragged_str))
  def testRaggedWithMapFn(self, factory_func):

    @def_function.function
    def create_using_map_fn(inputs):
      return map_fn.map_fn_v2(lambda x: x, inputs)

    t = factory_func()  # a ragged tensor containing a particular type
    if t.dtype == dtypes.string:
      self.skipTest('b/197903812: fix ragged tensor of string')
    result = self.evaluate(create_using_map_fn(t))
    self.assertAllEqual(self.evaluate(t).values, result.values)

  @parameterized.named_parameters(
      dict(testcase_name='Int', factory_func=_ragged_int),
      dict(testcase_name='String', factory_func=_ragged_str))
  def testRaggedWithDistributedDataset(self, factory_func):

    @def_function.function
    def _create_from_distributed_dataset(t):
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      ragged_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(ragged_ds)
      ds = iter(dist_dataset)
      return strategy.experimental_local_results(next(ds))

    t = factory_func()  # a ragged tensor containing a particular type
    if t.dtype == dtypes.string:
      self.skipTest('b/194439197: fix ragged tensor of string')
    result = _create_from_distributed_dataset(t)
    self.assertAllEqual(self.evaluate(t[0]), self.evaluate(result[0][0]))


if __name__ == '__main__':
  test.main()
