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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu
import os

from tensorflow.keras import layers
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops


class ReplicationNormaliseTest(test_util.TensorFlowTestCase):
  def testReplicationNormalise(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = gen_poputil_ops.ipu_replication_normalise(x)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True, replicated=True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      res = sess.run(y, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 0.5))
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*',
          'IpuReplicationNormalise/custom-call*/replication_normalise/Op/Divide',
          'switchControlBroadcast*/GlobalPre/Copy/OnTileCopy',
          'host-exchange-local-copy-',
          '/OnTileCopy',
          'Copy_XLA_Args*OnTileCopy',
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testReplicationNormaliseNotInplace(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      a = gen_poputil_ops.ipu_replication_normalise(x)
      b = a + x

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True, replicated=True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      res = sess.run(b, {x: np.ones([1, 4, 4, 2])})
      self.assertAllClose(res, np.full([1, 4, 4, 2], 1.5))
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*',
          'IpuReplicationNormalise/custom-call*/replication_normalise/Op/Divide',
          'switchControlBroadcast*/GlobalPre/Copy/OnTileCopy',
          'host-exchange-local-copy-',
          '/OnTileCopy',
          'Copy_XLA_Args*OnTileCopy',
          'add/add*/AddTo',
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
