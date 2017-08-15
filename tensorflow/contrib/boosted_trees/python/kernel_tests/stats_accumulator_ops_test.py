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
"""Test for checking stats accumulator related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.boosted_trees.python.ops import stats_accumulator_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class StatsAccumulatorScalarTest(test_util.TensorFlowTestCase):
  """Tests for scalar gradients and hessians accumulator."""

  def testSimpleAcculumator(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar())
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            gradients=[0.1, 0.3],
            hessians=[0.2, 0.4])
        op2 = accumulator.add(0, [1], [2], [0.1], [0.2])

      with ops.control_dependencies([op1, op2]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=0, next_stamp_token=1)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(num_updates, 2)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)], [0.2, 0.4])
      self.assertAllClose(result[(2, 3)], [0.3, 0.4])

  def testDropStaleUpdate(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar())
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            gradients=[0.1, 0.3],
            hessians=[0.2, 0.4])
        op2 = accumulator.add(
            stamp_token=-1,
            partition_ids=[1],
            feature_ids=[2],
            gradients=[0.1],
            hessians=[0.2])

      with ops.control_dependencies([op1, op2]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=0, next_stamp_token=1)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(num_updates, 1)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)], [0.1, 0.2])
      self.assertAllClose(result[(2, 3)], [0.3, 0.4])

  def testSerialize(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar())
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            gradients=[0.1, 0.3],
            hessians=[0.2, 0.4])

      with ops.control_dependencies([op1]):
        (stamp_token, num_updates, partition_1, feature_1, grads_1,
         hessians_1) = accumulator.serialize()
      # Make sure that the accumulator hasn't changed during serialization.
      with ops.control_dependencies([stamp_token]):
        num_updates_2, partition_2, feature_2, grads_2, hessians_2 = (
            accumulator.flush(stamp_token=0, next_stamp_token=1))
        (stamp_token, num_updates, partition_1, feature_1, grads_1, hessians_1,
         num_updates_2, partition_2, feature_2, grads_2, hessians_2) = sess.run(
             [
                 stamp_token, num_updates, partition_1, feature_1, grads_1,
                 hessians_1, num_updates_2, partition_2, feature_2, grads_2,
                 hessians_2
             ])

      result_1 = _AccumulatorResultToDict(partition_1, feature_1, grads_1,
                                          hessians_1)
      result_2 = _AccumulatorResultToDict(partition_2, feature_2, grads_2,
                                          hessians_2)
      self.assertEqual(num_updates, 1)
      self.assertEqual(num_updates_2, 1)
      self.assertEqual(len(result_1), 2)
      self.assertAllClose(result_1[(1, 2)], [0.1, 0.2])
      self.assertAllClose(result_1[(2, 3)], [0.3, 0.4])
      self.assertAllEqual(result_1, result_2)
      self.assertEqual(0, stamp_token)

  def testDeserialize(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar())
      with ops.control_dependencies([accumulator._create_op]):
        # These will be deleted due to deserialize call.
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            gradients=[0.1, 0.3],
            hessians=[0.2, 0.4])

      with ops.control_dependencies([op1]):
        deserialize = (accumulator.deserialize(
            stamp_token=2,
            num_updates=3,
            partition_ids=[3, 4],
            feature_ids=[5, 6],
            gradients=[0.4, 0.5],
            hessians=[0.6, 0.7]))
      with ops.control_dependencies([deserialize]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=2, next_stamp_token=3)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads,
                                        hessians)
      self.assertEqual(num_updates, 3)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(3, 5)], [0.4, 0.6])
      self.assertAllClose(result[(4, 6)], [0.5, 0.7])

  def testMakeSummary(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar())
      partition, feature, grads, hessians = accumulator._make_summary(
          partition_ids=[1, 2, 1],
          feature_ids=[2, 3, 2],
          gradients=[0.1, 0.3, 0.1],
          hessians=[0.2, 0.4, 0.2])
      partition, feature, grads, hessians = sess.run(
          [partition, feature, grads, hessians])
      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)], [0.2, 0.4])
      self.assertAllClose(result[(2, 3)], [0.3, 0.4])


class StatsAccumulatorTensorTest(test_util.TensorFlowTestCase):
  """Tests for tensor gradients and hessians accumulator."""

  def testSimpleAcculumator(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.TensorShape([2]),
          hessian_shape=tensor_shape.TensorShape([2, 2]))
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            # Two values for gradients,
            gradients=[[0.1, 0.1], [0.2, 0.2]],
            # A 2x2 matrix for each hessian.
            hessians=[[[0.01, 0.02], [0.03, 0.04]],
                      [[0.05, 0.06], [0.07, 0.08]]])
        op2 = accumulator.add(
            stamp_token=0,
            partition_ids=[1],
            feature_ids=[2],
            gradients=[[0.10, 0.11]],
            hessians=[[[0.011, 0.022], [0.033, 0.044]]])

      with ops.control_dependencies([op1, op2]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=0, next_stamp_token=1)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(num_updates, 2)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)][0], [0.20, 0.21])
      self.assertAllClose(result[(1, 2)][1], [[0.021, 0.042], [0.063, 0.084]])
      self.assertAllClose(result[(2, 3)][0], [0.2, 0.2])
      self.assertAllClose(result[(2, 3)][1], [[0.05, 0.06], [0.07, 0.08]])

  def testDropStaleUpdate(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.TensorShape([2]),
          hessian_shape=tensor_shape.TensorShape([2, 2]))
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            # Two values for gradients,
            gradients=[[0.1, 0.1], [0.2, 0.2]],
            # A 2x2 matrix for each hessian.
            hessians=[[[0.01, 0.02], [0.03, 0.04]],
                      [[0.05, 0.06], [0.07, 0.08]]])
        op2 = accumulator.add(
            stamp_token=-1,
            partition_ids=[1],
            feature_ids=[2],
            gradients=[[0.10, 0.11]],
            hessians=[[[0.011, 0.022], [0.033, 0.044]]])

      with ops.control_dependencies([op1, op2]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=0, next_stamp_token=1)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(num_updates, 1)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)][0], [0.1, 0.1])
      self.assertAllClose(result[(1, 2)][1], [[0.01, 0.02], [0.03, 0.04]])
      self.assertAllClose(result[(2, 3)][0], [0.2, 0.2])
      self.assertAllClose(result[(2, 3)][1], [[0.05, 0.06], [0.07, 0.08]])

  def testSerialize(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.TensorShape([2]),
          hessian_shape=tensor_shape.TensorShape([2, 2]))
      with ops.control_dependencies([accumulator._create_op]):
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            # Two values for gradients,
            gradients=[[0.1, 0.1], [0.2, 0.2]],
            # A 2x2 matrix for each hessian.
            hessians=[[[0.01, 0.02], [0.03, 0.04]],
                      [[0.05, 0.06], [0.07, 0.08]]])

      with ops.control_dependencies([op1]):
        (stamp_token, num_updates_1, partition_1, feature_1, grads_1,
         hessians_1) = accumulator.serialize()
      # Make sure that the accumulator hasn't changed during serialization.
      with ops.control_dependencies([stamp_token]):
        num_updates_2, partition_2, feature_2, grads_2, hessians_2 = (
            accumulator.flush(stamp_token=0, next_stamp_token=1))
        (stamp_token, num_updates_1, partition_1, feature_1, grads_1,
         hessians_1, num_updates_2, partition_2, feature_2, grads_2,
         hessians_2) = sess.run([
             stamp_token, num_updates_1, partition_1, feature_1, grads_1,
             hessians_1, num_updates_2, partition_2, feature_2, grads_2,
             hessians_2
         ])

      result_1 = _AccumulatorResultToDict(partition_1, feature_1, grads_1,
                                          hessians_1)
      result_2 = _AccumulatorResultToDict(partition_2, feature_2, grads_2,
                                          hessians_2)

      self.assertEqual(num_updates_1, 1)
      self.assertEqual(num_updates_2, 1)
      self.assertEqual(len(result_1), 2)
      self.assertAllClose(result_1[(1, 2)][0], [0.1, 0.1])
      self.assertAllClose(result_1[(1, 2)][1], [[0.01, 0.02], [0.03, 0.04]])
      self.assertAllClose(result_1[(2, 3)][0], [0.2, 0.2])
      self.assertAllClose(result_1[(2, 3)][1], [[0.05, 0.06], [0.07, 0.08]])

      self.assertAllEqual(result_1[1, 2][0], result_2[1, 2][0])
      self.assertAllEqual(result_1[1, 2][1], result_2[1, 2][1])
      self.assertAllEqual(result_1[2, 3][0], result_2[2, 3][0])
      self.assertAllEqual(result_1[2, 3][1], result_2[2, 3][1])

  def testDeserialize(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.TensorShape([2]),
          hessian_shape=tensor_shape.TensorShape([2, 2]))
      with ops.control_dependencies([accumulator._create_op]):
        # These will be deleted due to deserialize call.
        op1 = accumulator.add(
            stamp_token=0,
            partition_ids=[1, 2],
            feature_ids=[2, 3],
            # Two values for gradients,
            gradients=[[0.1, 0.1], [0.2, 0.2]],
            # A 2x2 matrix for each hessian.
            hessians=[[[0.01, 0.02], [0.03, 0.04]],
                      [[0.05, 0.06], [0.07, 0.08]]])

      with ops.control_dependencies([op1]):
        deserialize = accumulator.deserialize(
            stamp_token=2,
            num_updates=3,
            partition_ids=[3, 4],
            feature_ids=[4, 5],
            # Two values for gradients,
            gradients=[[0.3, 0.3], [0.5, 0.5]],
            # A 2x2 matrix for each hessian.
            hessians=[[[0.03, 0.04], [0.05, 0.06]], [[0.07, 0.08], [0.09,
                                                                    0.10]]])
      with ops.control_dependencies([deserialize]):
        num_updates, partition, feature, grads, hessians = accumulator.flush(
            stamp_token=2, next_stamp_token=3)
        num_updates, partition, feature, grads, hessians = sess.run(
            [num_updates, partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads,
                                        hessians)
      self.assertEqual(num_updates, 3)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(3, 4)][0], [0.3, 0.3])
      self.assertAllClose(result[(3, 4)][1], [[0.03, 0.04], [0.05, 0.06]])
      self.assertAllClose(result[(4, 5)][0], [0.5, 0.5])
      self.assertAllClose(result[(4, 5)][1], [[0.07, 0.08], [0.09, 0.10]])

  def testMakeSummary(self):
    with self.test_session() as sess:
      accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.TensorShape([2]),
          hessian_shape=tensor_shape.TensorShape([2, 2]))
      partition, feature, grads, hessians = accumulator._make_summary(
          partition_ids=[1, 2, 1],
          feature_ids=[2, 3, 2],
          # Two values for gradients,
          gradients=[[0.1, 0.1], [0.2, 0.2], [0.10, 0.11]],
          # A 2x2 matrix for each hessian.
          hessians=[[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]],
                    [[0.011, 0.022], [0.033, 0.044]]])
      partition, feature, grads, hessians = sess.run(
          [partition, feature, grads, hessians])

      result = _AccumulatorResultToDict(partition, feature, grads, hessians)
      self.assertEqual(len(result), 2)
      self.assertAllClose(result[(1, 2)][0], [0.20, 0.21])
      self.assertAllClose(result[(1, 2)][1], [[0.021, 0.042], [0.063, 0.084]])
      self.assertAllClose(result[(2, 3)][0], [0.2, 0.2])
      self.assertAllClose(result[(2, 3)][1], [[0.05, 0.06], [0.07, 0.08]])


def _AccumulatorResultToDict(partition, feature, grads, hessians):
  """Converts the inputs to a dictionary since the ordering changes."""
  return {(partition[i], feature[i]): (grads[i], hessians[i])
          for i in range(len(partition))}


if __name__ == "__main__":
  googletest.main()
