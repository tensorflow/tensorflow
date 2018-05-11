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
"""Tests for MultiWorkerMirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import multi_worker_strategy
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.contrib.distribute.python import strategy_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.training import server_lib


@test_util.with_c_api
class MultiWorkerStrategyTest(multi_worker_test_base.MultiWorkerTestBase,
                              strategy_test_lib.DistributionTestBase):

  def _get_distribution_strategy(self):
    return multi_worker_strategy.MultiWorkerMirroredStrategy(
        cluster=server_lib.ClusterSpec({
            'worker': ['/job:worker/task:0', '/job:worker/task:1']
        }),
        num_gpus_per_worker=context.num_gpus())

  def testMinimizeLossGraph(self):
    self._test_minimize_loss_graph(self._get_distribution_strategy())


class DeviceScopeTest(test.TestCase):
  """Test the device scope of MultiWorkerMirroredStrategy."""

  def testDeviceScope(self):
    with context.graph_mode():
      strategy = multi_worker_strategy.MultiWorkerMirroredStrategy(
          cluster={'worker': ['/job:worker/task:0', '/job:worker/task:1']},
          num_gpus_per_worker=context.num_gpus())
      with strategy.scope():
        a = constant_op.constant(1.)
        with ops.device('/cpu:0'):
          b = constant_op.constant(1.)
        self.assertEqual(a.device, '/job:worker/task:0')
        self.assertEqual(b.device, '/job:worker/task:0/device:CPU:0')


if __name__ == '__main__':
  test.main()
