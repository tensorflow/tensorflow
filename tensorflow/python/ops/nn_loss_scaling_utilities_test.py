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

from unittest import mock

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl_distribute
from tensorflow.python.platform import test as test_lib


class LossUtilitiesTest(test_lib.TestCase, parameterized.TestCase):

  def testComputeAverageLossGlobalBatchSize(self):
    per_example_loss = [1, 2, 3, 4, 5]
    loss = nn_impl_distribute.compute_average_loss(
        per_example_loss, global_batch_size=10
    )
    self.assertEqual(self.evaluate(loss), 1.5)

  def testComputeAverageLossGlobalBatchSize_BatchSizeNonScalar(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        ValueError, "global_batch_size must be scalar"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=[10]
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeFloat(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        TypeError, "global_batch_size must be an int"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=10.0
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeNegative(self):
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        errors_impl.InvalidArgumentError,
        "global_batch_size must be non-negative"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=-1
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeZero(self):
    per_example_loss = [1, 2, 3, 4, 5]
    loss = nn_impl_distribute.compute_average_loss(
        per_example_loss, global_batch_size=0
    )
    self.assertEqual(self.evaluate(loss), 0.0)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDefaultGlobalBatchSize(self, distribution):
    # Without strategy - num replicas = 1
    per_example_loss = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl_distribute.compute_average_loss(per_example_loss)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss, args=(per_example_loss,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDefaultGlobalBatchSizeEmptyBatch(self,
                                                             distribution):
    per_example_loss = constant_op.constant([], dtypes.float32)
    loss = nn_impl_distribute.compute_average_loss(per_example_loss)
    self.assertEqual(self.evaluate(loss), 0.0)

    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss, args=(per_example_loss,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), 0.0)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossSampleWeights(self, distribution):
    with distribution.scope():
      # Scalar sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([2.0, 4.0, 6.0],),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2. + 4. + 6.) * 2. / 3)

      # Per example sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([2.0, 4.0, 6.0],),
          kwargs={"sample_weight": [0.3, 0.5, 0.2]},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 4. * 0.5 + 6. * 0.2) / 3)

      # Time-step sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([[2.0, 0.5], [4.0, 1.0]],),
          kwargs={"sample_weight": [[0.3, 0.7], [0.2, 0.8]]},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 0.5 * 0.7 + 4. * 0.2 + 1. * 0.8) / 2)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossSampleWeightsEmptyBatch(self, distribution):
    empty_rank0 = constant_op.constant([], dtypes.float32)

    with distribution.scope():
      # Scalar sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(empty_rank0,),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), 0.0)

      # Per example sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(empty_rank0,),
          kwargs={"sample_weight": empty_rank0},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), 0.0)

  def testComputeAverageLossInvalidSampleWeights(self):
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors_impl.InvalidArgumentError)):
      nn_impl_distribute.compute_average_loss(
          [2.5, 6.2, 5.0], sample_weight=[0.2, 0.8], global_batch_size=10
      )

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDtype(self, distribution):
    with distribution.scope():
      per_example_loss = constant_op.constant([2., 4., 6.],
                                              dtype=dtypes.float64)
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(per_example_loss,),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertEqual(loss.dtype, dtypes.float64)

  def testComputeAverageLossInvalidRank(self):
    per_example_loss = constant_op.constant(2.)

    # Static rank
    with self.assertRaisesRegex(
        ValueError, "Invalid value passed for `per_example_loss`. "
        "Expected a tensor with at least rank 1."):
      nn_impl_distribute.compute_average_loss(per_example_loss)

    with context.graph_mode():
      # Dynamic rank
      per_example_loss = array_ops.placeholder(dtype=dtypes.float32)
      loss = nn_impl_distribute.compute_average_loss(per_example_loss)

      with self.cached_session() as sess:
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "Invalid value passed for `per_example_loss`. "
            "Expected a tensor with at least rank 1."):
          sess.run(loss, {per_example_loss: 2})

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossInCrossReplicaContext(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError,
          "You are calling `compute_average_loss` in cross replica context"):
        nn_impl_distribute.compute_average_loss([2, 3])

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testScaleRegularizationLoss(self, distribution):
    # Without strategy - num replicas = 1
    reg_losses = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl_distribute.scale_regularization_loss(reg_losses)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.scale_regularization_loss, args=(reg_losses,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testScaleRegularizationLossInCrossReplicaContext(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError, "You are calling `scale_regularization_loss` in "
          "cross replica context"):
        nn_impl_distribute.scale_regularization_loss([2, 3])

# ==============================================================================
# REGRESSION TESTS FOR COMPUTE_AVERAGE_LOSS XLA CONTEXT MASKING
# ==============================================================================


class TestGetNumReplicasInSync(test_lib.TestCase):
  """Unit tests for the _get_num_replicas_in_sync helper."""

  def test_replica_context_takes_priority(self):
    """ReplicaContext.num_replicas_in_sync wins over strategy stack."""
    replica_ctx = mock.MagicMock()
    replica_ctx.num_replicas_in_sync = 4
    strategy = mock.MagicMock()
    strategy.num_replicas_in_sync = 8
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=replica_ctx,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib, "has_strategy", return_value=True
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_strategy",
            return_value=strategy,
        ),
    ):
      self.assertEqual(nn_impl_distribute._get_num_replicas_in_sync(), 4)

  def test_replica_context_returned_when_strategy_stack_absent(self):
    """ReplicaContext path works when has_strategy is False."""
    replica_ctx = mock.MagicMock()
    replica_ctx.num_replicas_in_sync = 2
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=replica_ctx,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=False,
        ),
    ):
      self.assertEqual(nn_impl_distribute._get_num_replicas_in_sync(), 2)

  def test_strategy_fallback_used_when_no_replica_context(self):
    """Falls back to strategy stack when get_replica_context returns None."""
    strategy = mock.MagicMock()
    strategy.num_replicas_in_sync = 6
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=None,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=True,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_strategy",
            return_value=strategy,
        ),
    ):
      self.assertEqual(nn_impl_distribute._get_num_replicas_in_sync(), 6)

  def test_default_returns_one_with_no_context(self):
    """Returns 1 when neither replica context nor strategy is active."""
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=None,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=False,
        ),
    ):
      self.assertEqual(nn_impl_distribute._get_num_replicas_in_sync(), 1)

  def test_invalid_replicas_in_replica_context_raises(self):
    """Raises ValueError when ReplicaContext.num_replicas_in_sync < 1."""
    bad_ctx = mock.MagicMock()
    bad_ctx.num_replicas_in_sync = 0
    with mock.patch.object(
        nn_impl_distribute.distribute_lib,
        "get_replica_context",
        return_value=bad_ctx,
    ):
      with self.assertRaises(ValueError):
        nn_impl_distribute._get_num_replicas_in_sync()

  def test_invalid_replicas_in_strategy_raises(self):
    """Raises ValueError when Strategy.num_replicas_in_sync < 1."""
    bad_strategy = mock.MagicMock()
    bad_strategy.num_replicas_in_sync = -1
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=None,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=True,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_strategy",
            return_value=bad_strategy,
        ),
    ):
      with self.assertRaises(ValueError):
        nn_impl_distribute._get_num_replicas_in_sync()


class TestComputeAverageLossXLAContextMasking(test_lib.TestCase):
  """Integration tests verifying compute_average_loss under XLA context masking.

  These tests simulate the scenario where jit_compile=True causes the
  thread-local strategy_stack to be invisible while the ReplicaContext is
  still populated.
  """

  def test_xla_mask_simulation_returns_correct_replica_count(self):
    """Simulates XLA masking: replica ctx visible, strategy stack absent."""
    replica_ctx = mock.MagicMock()
    replica_ctx.num_replicas_in_sync = 2
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=replica_ctx,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=False,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "in_cross_replica_context",
            return_value=False,
        ),
    ):
      per_example_loss = constant_op.constant([3.0, 3.0])
      result = nn_impl_distribute.compute_average_loss(
          per_example_loss,
      )
    # sum=6, replicas=2, batch=2, global=4 -> 6/4=1.5
    self.assertAllClose(self.evaluate(result), 1.5)

  def test_no_context_falls_back_to_single_replica(self):
    """Without any distribution context, single-replica."""
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=None,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=False,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "in_cross_replica_context",
            return_value=False,
        ),
    ):
      per_example_loss = constant_op.constant([2.0, 4.0])
      result = nn_impl_distribute.compute_average_loss(
          per_example_loss,
      )
    # sum=6, replicas=1, batch=2, global=2 -> 6/2=3.0
    self.assertAllClose(self.evaluate(result), 3.0)

  def test_replica_ctx_priority_over_strategy(self):
    """Replica context takes priority over strategy stack."""
    replica_ctx = mock.MagicMock()
    replica_ctx.num_replicas_in_sync = 4
    strategy = mock.MagicMock()
    strategy.num_replicas_in_sync = 8
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=replica_ctx,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=True,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_strategy",
            return_value=strategy,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "in_cross_replica_context",
            return_value=False,
        ),
    ):
      per_example_loss = constant_op.constant([1.0, 1.0])
      result = nn_impl_distribute.compute_average_loss(
          per_example_loss,
      )
    # sum=2, replicas=4, batch=2, global=8 -> 2/8=0.25
    self.assertAllClose(self.evaluate(result), 0.25)


class TestScaleRegularizationLossXLAContextMasking(
    test_lib.TestCase,
):
  """Verifies scale_regularization_loss under XLA masking."""

  def test_scale_regularization_loss_xla_mask(self):
    """XLA mask: replica ctx visible, strategy absent."""
    replica_ctx = mock.MagicMock()
    replica_ctx.num_replicas_in_sync = 2
    with (
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "get_replica_context",
            return_value=replica_ctx,
        ),
        mock.patch.object(
            nn_impl_distribute.distribute_lib,
            "has_strategy",
            return_value=False,
        ),
    ):
      reg_loss = constant_op.constant(4.0)
      result = nn_impl_distribute.scale_regularization_loss(
          reg_loss,
      )
    # 4.0 / 2 = 2.0
    self.assertAllClose(self.evaluate(result), 2.0)


if __name__ == "__main__":
  test_lib.main()
