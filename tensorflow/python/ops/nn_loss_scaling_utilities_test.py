# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for loss scaling utilities in tensorflow.ops.nn."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test as test_lib


class LossUtilitiesTest(test_lib.TestCase, parameterized.TestCase):

  def setUp(self):
    test_util.set_logical_devices_to_at_least("CPU", 3)
    super(LossUtilitiesTest, self).setUp()

  def testComputeAverageLossGlobalBatchSize(self):
    per_example_loss = [1, 2, 3, 4, 5]
    loss = nn_impl.compute_average_loss(per_example_loss, global_batch_size=10)
    self.assertEqual(self.evaluate(loss), 1.5)

  def testComputeAverageLossGlobalBatchSize_BatchSizeNonScalar(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        ValueError, "global_batch_size must be scalar"):
      nn_impl.compute_average_loss(per_example_loss, global_batch_size=[10])

  def testComputeAverageLossGlobalBatchSize_BatchSizeFloat(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        TypeError, "global_batch_size must be an int"):
      nn_impl.compute_average_loss(per_example_loss, global_batch_size=10.0)

  def testComputeAverageLossGlobalBatchSize_BatchSizeNegative(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        errors_impl.InvalidArgumentError, "global_batch_size must be positive"):
      nn_impl.compute_average_loss(per_example_loss, global_batch_size=-1)

  def testComputeAverageLossGlobalBatchSize_BatchSizeZero(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        errors_impl.InvalidArgumentError, "global_batch_size must be positive"):
      nn_impl.compute_average_loss(per_example_loss, global_batch_size=0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testComputeAverageLossDefaultGlobalBatchSize(self, distribution):
    # Without strategy - num replicas = 1
    per_example_loss = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl.compute_average_loss(per_example_loss)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl.compute_average_loss, args=(per_example_loss,))
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testComputeAverageLossSampleWeights(self, distribution):
    with distribution.scope():
      # Scalar sample weight
      per_replica_losses = distribution.run(
          nn_impl.compute_average_loss,
          args=([2., 4., 6.],),
          kwargs={"sample_weight": 2})
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2. + 4. + 6.) * 2. / 3)

      # Per example sample weight
      per_replica_losses = distribution.run(
          nn_impl.compute_average_loss,
          args=([2., 4., 6.],),
          kwargs={"sample_weight": [0.3, 0.5, 0.2]})
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 4. * 0.5 + 6. * 0.2) / 3)

      # Time-step sample weight
      per_replica_losses = distribution.run(
          nn_impl.compute_average_loss,
          args=([[2., 0.5], [4., 1.]],),
          kwargs={"sample_weight": [[0.3, 0.7], [0.2, 0.8]]})
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 0.5 * 0.7 + 4. * 0.2 + 1. * 0.8) / 2)

  def testComputeAverageLossInvalidSampleWeights(self):
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors_impl.InvalidArgumentError)):
      nn_impl.compute_average_loss([2.5, 6.2, 5.],
                                   sample_weight=[0.2, 0.8],
                                   global_batch_size=10)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testComputeAverageLossDtype(self, distribution):
    with distribution.scope():
      per_example_loss = constant_op.constant([2., 4., 6.],
                                              dtype=dtypes.float64)
      per_replica_losses = distribution.run(
          nn_impl.compute_average_loss,
          args=(per_example_loss,),
          kwargs={"sample_weight": 2})
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertEqual(loss.dtype, dtypes.float64)

  def testComputeAverageLossInvalidRank(self):
    per_example_loss = constant_op.constant(2)

    # Static rank
    with self.assertRaisesRegex(
        ValueError, "Invalid value passed for `per_example_loss`. "
        "Expected a tensor with at least rank 1."):
      nn_impl.compute_average_loss(per_example_loss)

    with context.graph_mode():
      # Dynamic rank
      per_example_loss = array_ops.placeholder(dtype=dtypes.float32)
      loss = nn_impl.compute_average_loss(per_example_loss)

      with self.cached_session() as sess:
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "Invalid value passed for `per_example_loss`. "
            "Expected a tensor with at least rank 1."):
          sess.run(loss, {per_example_loss: 2})

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testComputeAverageLossInCrossReplicaContext(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError,
          "You are calling `compute_average_loss` in cross replica context"):
        nn_impl.compute_average_loss([2, 3])

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testScaleRegularizationLoss(self, distribution):
    # Without strategy - num replicas = 1
    reg_losses = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl.scale_regularization_loss(reg_losses)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl.scale_regularization_loss, args=(reg_losses,))
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testScaleRegularizationLossInCrossReplicaContext(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError, "You are calling `scale_regularization_loss` in "
          "cross replica context"):
        nn_impl.scale_regularization_loss([2, 3])


if __name__ == "__main__":
  test_lib.main()
