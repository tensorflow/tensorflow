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

import fnmatch
import json
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.framework import errors
from tensorflow.contrib.ipu.python import popops_cross_replica_sum

class ReplicatedGraphTest(test_util.TensorFlowTestCase):

  def testCreateSimpleReplicatedGraph(self):
    def my_graph(inp):
      with ops.device("/device:IPU:0"):
        x = inp + inp

        return [popops_cross_replica_sum.cross_replica_sum(x)]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [4], name="data")

    out = ipu_compiler.compile(my_graph, [inp])

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, number_of_replicas=2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      data = np.ones([4])
      fd = {inp: data}

      result = sess.run(out, fd)

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * data)

  def testCreateSimpleReplicatedGraphVariable(self):
    def my_graph():
      with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          x = variable_scope.get_variable("x", dtype=np.float32, shape=[4],
            initializer=init_ops.constant_initializer(10.0))
        x = x + x
        return [popops_cross_replica_sum.cross_replica_sum(x)]

    out = ipu_compiler.compile(my_graph, [])

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, number_of_replicas=2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(out, {})

      # Test that the output is just the input
      self.assertAllClose(result[0], 4 * np.full([4], 10.0))

if __name__ == "__main__":
    googletest.main()
