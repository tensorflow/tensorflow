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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


# TODO(b/143286947): expand the test to cover fault tolerance and elasticity
class MultiWorkerContinuousRunTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['eager']))
  def testAllReduceContinuousRun(self, mode):
    num_workers = 5
    tensor_shape = [2, 2]
    local_device = '/device:CPU:0'
    if config.list_physical_devices('GPU'):
      local_device = '/device:GPU:0'

    def worker_step_fn():
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_id = tf_config['task']['index']

      @def_function.function
      def run_reduce():
        with ops.device(local_device):
          t_in = array_ops.ones(tensor_shape) * worker_id
          return strategy.reduce(reduce_util.ReduceOp.MEAN, t_in, axis=None)

      t_out = run_reduce()
      # Element values from the workers are
      #     0, 1, ..., (num_workers - 1)
      expected_mean = (num_workers - 1) / 2
      expected_out = np.ones(tensor_shape) * expected_mean
      self.assertAllClose(t_out, expected_out)

    def worker_fn():
      gpus = config.list_physical_devices('GPU')
      if gpus:
        # Set virtual GPU with memory limit of 64MB so that multiple worker
        # processes can share the physical GPU
        config.set_logical_device_configuration(
            gpus[0], [context.LogicalDeviceConfiguration(64)])
      for _ in range(100):
        worker_step_fn()

    multi_process_runner.run(
        worker_fn,
        cluster_spec=test_base.create_cluster_spec(num_workers=num_workers))


if __name__ == '__main__':
  multi_process_runner.test_main()
