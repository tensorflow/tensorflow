# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for cross_device_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops


class IndexedSlicesUtilsTest(test.TestCase, parameterized.TestCase):

  def _assert_values_equal(self, left, right):
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  @test_util.run_in_graph_and_eager_modes
  def testAggregateTensors(self):
    t0 = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    t1 = constant_op.constant([[0., 0.], [5, 6], [7., 8.]])
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
    self._assert_values_equal(total, result)

  @test_util.run_in_graph_and_eager_modes
  def testAggregateIndexedSlices(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
    self.assertIsInstance(result, ops.IndexedSlices)
    self._assert_values_equal(total, result)

  @test_util.run_in_graph_and_eager_modes
  def testDivideTensor(self):
    t = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    n = 2
    expected = constant_op.constant([[0.5, 1.], [0, 0], [1.5, 2.]])
    result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
    self._assert_values_equal(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testDivideIndexedSlices(self):
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    n = 2
    expected = constant_op.constant([[0.5, 1.], [0, 0], [1.5, 2.]])
    result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
    self.assertIsInstance(result, ops.IndexedSlices)
    self._assert_values_equal(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testIsIndexedSlices(self):
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    self.assertTrue(cross_device_utils.contains_indexed_slices(t))

  @test_util.run_in_graph_and_eager_modes
  def testContainsIndexedSlices_List(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    self.assertTrue(cross_device_utils.contains_indexed_slices([t0, t1]))

  @test_util.run_in_graph_and_eager_modes
  def testContainsIndexedSlices_Tuple(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    self.assertTrue(cross_device_utils.contains_indexed_slices((t0, t1)))

  @test_util.run_in_graph_and_eager_modes
  def testContainsIndexedSlices_PerReplica(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    device_map = value_lib.ReplicaDeviceMap(("/gpu:0", "/cpu:0"))
    per_replica = value_lib.PerReplica(device_map, (t0, t1))
    self.assertTrue(cross_device_utils.contains_indexed_slices(per_replica))

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testCopyTensor(self):
    with ops.device("/cpu:0"):
      t = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    destination = "/gpu:0"
    result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        t, destination)

    self._assert_values_equal(t, result)
    self.assertEqual(device_util.resolve(destination),
                     device_util.resolve(result.device))

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testCopyIndexedSlices(self):
    with ops.device("/cpu:0"):
      t = math_ops._as_indexed_slices(
          constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    destination = "/gpu:0"
    result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        t, destination)

    self.assertIsInstance(result, ops.IndexedSlices)
    self._assert_values_equal(t, result)
    self.assertEqual(device_util.resolve(destination),
                     device_util.resolve(result.device))


if __name__ == "__main__":
  test.main()
