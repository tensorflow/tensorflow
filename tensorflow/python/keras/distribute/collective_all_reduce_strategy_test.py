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
"""Tests for CollectiveAllReduceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.mixed_precision import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import training_util
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scale_optimizer
from tensorflow.python.training.server_lib import ClusterSpec


def create_test_objects(cluster_spec=None,
                        task_type=None,
                        task_id=None,
                        num_gpus=None):
  sess_config = config_pb2.ConfigProto()
  if num_gpus is None:
    num_gpus = len(tf_config.list_logical_devices('GPU'))

  if cluster_spec and task_type and task_id is not None:
    cluster_resolver = SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type=task_type,
        task_id=task_id,
        num_accelerators={'GPU': num_gpus})
    target = 'grpc://' + cluster_spec[task_type][task_id]
  else:
    cluster_resolver = SimpleClusterResolver(
        ClusterSpec({}), num_accelerators={'GPU': num_gpus})
    target = ''

  strategy = mwms_lib.CollectiveAllReduceStrategy(
      cluster_resolver=cluster_resolver)
  sess_config = strategy.update_config_proto(sess_config)

  return strategy, target, sess_config


class CollectiveAllReduceStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):

  collective_key_base = 0

  def setUp(self):
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    mwms_lib.CollectiveAllReduceStrategy._collective_key_base += 100000
    super(CollectiveAllReduceStrategyTestBase, self).setUp()

  def _get_test_object(self, task_type, task_id, num_gpus=0):
    strategy, target, session_config = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id,
        num_gpus=num_gpus)
    return strategy, target, session_config

  def _test_complex_model(self, task_type, task_id, num_gpus):
    d, master_target, config = self._get_test_object(task_type, task_id,
                                                     num_gpus)

    def model_fn():
      """Mnist model with synthetic input."""
      data_format = 'channels_last'
      input_shape = [28, 28, 1]
      l = layers
      max_pool = l.MaxPooling2D((2, 2), (2, 2),
                                padding='same',
                                data_format=data_format)
      model = sequential.Sequential([
          l.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=nn.relu), max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=nn.relu), max_pool,
          l.Flatten(),
          l.Dense(1024, activation=nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])
      image = random_ops.random_uniform([2, 28, 28])
      label = random_ops.random_uniform([2, 1], maxval=10, dtype=dtypes.int32)
      logits = model(image, training=True)
      # TODO(yuefengz): make loss a callable for eager mode.
      loss = losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
      optimizer = adam.AdamOptimizer(learning_rate=1e-4)
      train_op = optimizer.minimize(loss,
                                    training_util.get_or_create_global_step())
      return train_op

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      with d.scope():
        train_op = d.extended.call_for_each_replica(model_fn)
        train_op = d.group(d.experimental_local_results(train_op))

      sess.run(variables.global_variables_initializer())
      sess.run(train_op)

  def _test_mixed_precision(self, task_type, task_id, num_gpus):
    """Tests mixed precision works with the CollectiveAllReduceStrategy.

    This tests:
      1. Variables are in float32, by running with a small enough learning rate
         that if the variables are float16, their values wouldn't change when
         gradients are applied.
      2. The loss scale is doubled if there are no NaNs.
      3. The loss scale is halved if the first worker has a NaN, even if the
         other works do not have NaNs.

    Args:
      task_type: A string, such as "worker", indicating the type of the replica.
      task_id: Zero-indexed ID of the task.
      num_gpus: The number of GPUs to use.
    """
    d, master_target, config = self._get_test_object(task_type, task_id,
                                                     num_gpus)
    # Should be set to mixed_float16 by caller.
    self.assertEqual(policy.global_policy().name, 'mixed_float16')

    with ops.Graph().as_default(), \
         self.cached_session(config=config,
                             target=master_target) as sess:
      # The loss on the first worker is multiplied by this value. Allows
      # testing the first worker having NaN loss and gradients while keeping the
      # other workers' losses and gradients finite.
      loss_multiplier_for_first_worker = variables.Variable(
          1., dtype='float16', trainable=False)
      with d.scope():
        model = sequential.Sequential([
            mp_test_util.MultiplyLayer(assert_type=dtypes.float16,
                                       input_shape=(1,)),
        ])
        loss_scale = loss_scale_module.DynamicLossScale(2 ** 10,
                                                        increment_period=1)
        def model_fn():
          """Simple model to test mixed precision."""
          x = np.ones((1, 1))
          loss = model(x, training=True)

          if ((task_type == 'worker' and task_id == 0) or
              task_type is task_id is None):
            loss *= loss_multiplier_for_first_worker
          # Learning rate is small enough that if applied to a float16 variable,
          # the variable will not change. So this tests the learning rate is not
          # applied to a float16 value, but instead the float32 variable.
          optimizer = gradient_descent.GradientDescentOptimizer(2 ** -14)
          optimizer = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
              optimizer, loss_scale)
          train_op = optimizer.minimize(
              loss, training_util.get_or_create_global_step())
          return train_op

        train_op = d.extended.call_for_each_replica(model_fn)
        train_op = d.group(d.experimental_local_results(train_op))

      sess.run(variables.global_variables_initializer())
      sess.run(train_op)

      (var,) = model.trainable_weights
      # Variable starts at 1. Each worker's gradient is 2 ** -14, the learning
      # rate, and each worker's gradient will be subtracted from the variable.
      expected = 1 - d.num_replicas_in_sync * 2 ** -14
      self.assertEqual(sess.run(var), expected)
      # Loss scale should double, as are gradients are finite.
      self.assertEqual(sess.run(loss_scale()), 2 ** 11)

      # Set the first worker to have NaN loss and gradients.
      sess.run(loss_multiplier_for_first_worker.assign(float('NaN')))
      sess.run(train_op)
      # Variable should not change, since first worker had NaN
      self.assertEqual(sess.run(var), expected)
      # Loss scale should halve due to NaN
      self.assertEqual(sess.run(loss_scale()), 2 ** 10)


class DistributedCollectiveAllReduceStrategyTest(
    CollectiveAllReduceStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0)

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testComplexModel(self, required_gpus):
    self._run_between_graph_clients(
        self._test_complex_model, self._cluster_spec, num_gpus=required_gpus)

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  @testing_utils.enable_v2_dtype_behavior
  def testMixedPrecision(self, required_gpus):
    if test_util.is_xla_enabled():
      self.skipTest('Test gets NaNs with XLA')
    with policy.policy_scope('mixed_float16'):
      self._run_between_graph_clients(
          self._test_mixed_precision,
          self._cluster_spec,
          num_gpus=required_gpus)


class DistributedCollectiveAllReduceStrategyTestWithChief(
    CollectiveAllReduceStrategyTestBase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 3 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0, has_chief=True)

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testComplexModel(self, required_gpus):
    self._run_between_graph_clients(
        self._test_complex_model, self._cluster_spec, num_gpus=required_gpus)

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  @testing_utils.enable_v2_dtype_behavior
  def testMixedPrecision(self, required_gpus):
    if test_util.is_xla_enabled():
      return  # Test gets NaNs with XLA
    with policy.policy_scope('mixed_float16'):
      self._run_between_graph_clients(
          self._test_mixed_precision,
          self._cluster_spec,
          num_gpus=required_gpus)


class LocalCollectiveAllReduceStrategy(
    CollectiveAllReduceStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[2, 4]))
  def testComplexModel(self, required_gpus):
    self._test_complex_model(None, None, required_gpus)

  @ds_combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[2, 4]))
  @testing_utils.enable_v2_dtype_behavior
  def testMixedPrecision(self, required_gpus):
    with policy.policy_scope('mixed_float16'):
      self._test_mixed_precision(None, None, required_gpus)

# TODO(b/170360740): Timeout in OSS
if not multi_process_runner.is_oss():

  @ds_combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=['eager']))
  class DistributedCollectiveAllReduceStrategyEagerTest(test.TestCase,
                                                        parameterized.TestCase):

    def testFitWithoutStepsPerEpochPartialBatch(self, strategy):

      def _model_fn():
        x = layers.Input(shape=(1,), name='input')
        y = layers.Dense(1, name='dense')(x)
        model = training.Model(x, y)
        return model

      def _get_dataset():
        inputs = array_ops.expand_dims_v2(
            constant_op.constant(range(10)), axis=1)
        targets = array_ops.expand_dims_v2(
            constant_op.constant(range(10)), axis=1)
        # Make global batch size 12 for 2 replicas and a non-repeated dataset
        # with 10 elements so that we have partial batch
        dataset = dataset_ops.Dataset.from_tensor_slices(
            (inputs, targets)).batch(
                12, drop_remainder=False)
        return dataset

      with strategy.scope():
        optimizer_fn = gradient_descent_keras.SGD
        optimizer = optimizer_fn(0.001)
        model = _model_fn()
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)
      dataset = _get_dataset()
      kernel_before = model.get_weights()[0][0]
      model.fit(dataset, epochs=10)
      kernel_after = model.get_weights()[0][0]
      self.assertNotEqual(kernel_before, kernel_after)
      self.assertGreater(abs(kernel_before - 1), abs(kernel_after - 1))


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
