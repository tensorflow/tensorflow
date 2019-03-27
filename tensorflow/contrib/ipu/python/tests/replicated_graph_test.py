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
from tensorflow.python.framework import constant_op
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
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ops import gen_array_ops
from tensorflow.contrib.ipu import ipu_infeed_queue
from tensorflow.contrib.ipu import ipu_outfeed_queue
from tensorflow.contrib.ipu import loops


def create_increasing_dataset(value, shape=[4, 4], dtype=np.float32):
  def _get_one_input(data):
    return math_ops.cast(
        gen_array_ops.broadcast_to(data, shape=shape), dtype=dtype)

  dataset = Dataset.range(value).repeat().map(_get_one_input)
  return dataset


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
          x = variable_scope.get_variable(
              "x",
              dtype=np.float32,
              shape=[4],
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

  def testCreateSimpleReplicatedInfeedOutfeed(self):
    shape = [2]
    dataset = create_increasing_dataset(3, shape=shape)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, replication_factor=2)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(replication_factor=2)

    def body(v, x):
      v = popops_cross_replica_sum.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue(v)
      return (v, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, number_of_replicas=2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result.shape[0], 2)
      self.assertAllClose(outfed_result[0][0], outfed_result[0][1])
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result[1][0], outfed_result[1][1])
      self.assertAllClose(outfed_result[1][0], np.broadcast_to(4, shape))

      self.assertAllClose(outfed_result[2][0], outfed_result[2][1])
      self.assertAllClose(outfed_result[2][0], np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result[3][0], outfed_result[3][1])
      self.assertAllClose(outfed_result[3][0], np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result[4][0], outfed_result[4][1])
      self.assertAllClose(outfed_result[4][0], np.broadcast_to(48, shape))

  def testCreateSimpleReplicatedInfeedOutfeedTuple(self):
    shape = [2]
    dataset = create_increasing_dataset(3, shape=shape)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, replication_factor=2)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(replication_factor=2)

    def body(v, x):
      out = popops_cross_replica_sum.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue((v, out))
      return (out, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, number_of_replicas=2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result[0].shape[0], 2)
      self.assertTrue(outfed_result[1].shape[0], 2)
      self.assertAllClose(outfed_result[0][0][0], outfed_result[0][0][1])
      self.assertAllClose(outfed_result[0][0][0], np.broadcast_to(0, shape))
      self.assertAllClose(outfed_result[1][0][0], outfed_result[1][0][1])
      self.assertAllClose(outfed_result[1][0][0], np.broadcast_to(1, shape))

      self.assertAllClose(outfed_result[0][1][0], outfed_result[0][1][1])
      self.assertAllClose(outfed_result[0][1][0], np.broadcast_to(1, shape))
      self.assertAllClose(outfed_result[1][1][0], outfed_result[1][1][1])
      self.assertAllClose(outfed_result[1][1][0], np.broadcast_to(4, shape))

      self.assertAllClose(outfed_result[0][2][0], outfed_result[0][2][1])
      self.assertAllClose(outfed_result[0][2][0], np.broadcast_to(4, shape))
      self.assertAllClose(outfed_result[1][2][0], outfed_result[1][2][1])
      self.assertAllClose(outfed_result[1][2][0], np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result[0][3][0], outfed_result[0][3][1])
      self.assertAllClose(outfed_result[0][3][0], np.broadcast_to(11, shape))
      self.assertAllClose(outfed_result[1][3][0], outfed_result[1][3][1])
      self.assertAllClose(outfed_result[1][3][0], np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result[0][4][0], outfed_result[0][4][1])
      self.assertAllClose(outfed_result[0][4][0], np.broadcast_to(23, shape))
      self.assertAllClose(outfed_result[1][4][0], outfed_result[1][4][1])
      self.assertAllClose(outfed_result[1][4][0], np.broadcast_to(48, shape))

  def testCreateSimpleReplicatedInfeedOutfeedDict(self):
    shape = [2]
    dataset = create_increasing_dataset(3, shape=shape)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, replication_factor=2)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(replication_factor=2)

    def body(v, x):
      out = popops_cross_replica_sum.cross_replica_sum(v + x)
      outfeed = outfeed_queue.enqueue({"last": v, "this": out})
      return (out, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, number_of_replicas=2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(48, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result["last"].shape[0], 2)
      self.assertTrue(outfed_result["this"].shape[0], 2)
      self.assertAllClose(outfed_result["last"][0][0],
                          outfed_result["last"][0][1])
      self.assertAllClose(outfed_result["last"][0][0], np.broadcast_to(
          0, shape))
      self.assertAllClose(outfed_result["this"][0][0],
                          outfed_result["this"][0][1])
      self.assertAllClose(outfed_result["this"][0][0], np.broadcast_to(
          1, shape))

      self.assertAllClose(outfed_result["last"][1][0],
                          outfed_result["last"][1][1])
      self.assertAllClose(outfed_result["last"][1][0], np.broadcast_to(
          1, shape))
      self.assertAllClose(outfed_result["this"][1][0],
                          outfed_result["this"][1][1])
      self.assertAllClose(outfed_result["this"][1][0], np.broadcast_to(
          4, shape))

      self.assertAllClose(outfed_result["last"][2][0],
                          outfed_result["last"][2][1])
      self.assertAllClose(outfed_result["last"][2][0], np.broadcast_to(
          4, shape))
      self.assertAllClose(outfed_result["this"][2][0],
                          outfed_result["this"][2][1])
      self.assertAllClose(outfed_result["this"][2][0],
                          np.broadcast_to(11, shape))

      self.assertAllClose(outfed_result["last"][3][0],
                          outfed_result["last"][3][1])
      self.assertAllClose(outfed_result["last"][3][0],
                          np.broadcast_to(11, shape))
      self.assertAllClose(outfed_result["this"][3][0],
                          outfed_result["this"][3][1])
      self.assertAllClose(outfed_result["this"][3][0],
                          np.broadcast_to(23, shape))

      self.assertAllClose(outfed_result["last"][4][0],
                          outfed_result["last"][4][1])
      self.assertAllClose(outfed_result["last"][4][0],
                          np.broadcast_to(23, shape))
      self.assertAllClose(outfed_result["this"][4][0],
                          outfed_result["this"][4][1])
      self.assertAllClose(outfed_result["this"][4][0],
                          np.broadcast_to(48, shape))


if __name__ == "__main__":
  googletest.main()
