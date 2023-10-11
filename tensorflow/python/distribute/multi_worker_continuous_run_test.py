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
"""Tests for continuous runs using cross-worker collective ops."""

import json
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn


# TODO(b/151232436): This test doesn't work with check health enabled because it
# relies on barrier around creating strategies. Check health performs
# communications inside strategy constructor, which makes the barrier
# ineffective.
CollectiveAllReduceExtended = (
    collective_all_reduce_strategy.CollectiveAllReduceExtended)
CollectiveAllReduceExtended._enable_check_health = False


NUM_WORKERS = 5


# TODO(b/143286947): expand the test to cover fault tolerance and elasticity
class MultiWorkerContinuousRunTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MultiWorkerContinuousRunTest, self).setUp()
    self._maybe_setup_gpus(setup=True)

  def _maybe_setup_gpus(self, setup=False):
    self._gpus = config.list_physical_devices('GPU')
    self._local_device = '/device:GPU:0' if self._gpus else '/device:CPU:0'

    if self._gpus and not setup:
      # Set virtual GPU with memory limit of 64MB so that multiple worker
      # processes can share the physical GPU
      config.set_logical_device_configuration(
          self._gpus[0], [context.LogicalDeviceConfiguration(64)])

  @combinations.generate(combinations.combine(mode=['eager']))
  def testAllReduceContinuousRun(self, mode):
    tensor_shape = [2, 2]

    def worker_step_fn(worker_id):
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      # Make sure the processeses are in sync after updating the cluster
      multi_process_runner.get_barrier().wait()

      @def_function.function
      def run_reduce():
        with ops.device(self._local_device):
          t_in = array_ops.ones(tensor_shape) * worker_id
          return strategy.reduce(reduce_util.ReduceOp.MEAN, t_in, axis=None)

      t_out = run_reduce()
      # Element values from the workers are
      #     0, 1, ..., (NUM_WORKERS - 1)
      expected_mean = (NUM_WORKERS - 1) / 2
      expected_out = np.ones(tensor_shape) * expected_mean
      self.assertAllClose(t_out, expected_out)

    def worker_fn():
      self._maybe_setup_gpus()
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_id = tf_config['task']['index']
      for _ in range(20):
        worker_step_fn(worker_id)

    with test_util.skip_if_error(self, errors_impl.UnavailableError):
      multi_process_runner.run(
          worker_fn,
          cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))

  @combinations.generate(combinations.combine(mode=['eager']))
  def testVariableInitializationWithChangingShape(self, mode):

    def worker_step_fn(worker_id, num_dims):
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      # Make sure the processeses are in sync after updating the cluster
      multi_process_runner.get_barrier().wait()
      tensor_shape = [2] * num_dims

      def variable_fn():
        with ops.device(self._local_device):
          # The initial value will be broadcasted from worker 0 to others.
          initial_value = (array_ops.ones(tensor_shape) if worker_id == 0 else
                           array_ops.zeros(tensor_shape))
          var = variable_scope.get_variable(name='x', initializer=initial_value)
          return array_ops.identity(var)

      t_out = strategy.extended.call_for_each_replica(variable_fn)
      expected_out = np.ones(tensor_shape)
      self.assertAllClose(t_out, expected_out)

    def worker_fn():
      self._maybe_setup_gpus()
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_id = tf_config['task']['index']
      for i in range(20):
        worker_step_fn(worker_id, num_dims=(i + 1))

    with test_util.skip_if_error(self, errors_impl.UnavailableError):
      multi_process_runner.run(
          worker_fn,
          cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))


@_REGISTER_DECORATOR(MultiWorkerContinuousRunTest)
def _save_test_case(pickler, obj):
  def reconstruct(*args, **kwargs):
    del args, kwargs
    return MultiWorkerContinuousRunTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  multi_process_runner.test_main()
