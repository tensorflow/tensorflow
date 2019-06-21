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
import test_util as tu

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as internal_tu
from tensorflow.contrib import ipu
from tensorflow.keras import layers
from tensorflow.python.client import session as sl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
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
from tensorflow.contrib.ipu import popops_cross_replica_sum
from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib.ipu import ipu_optimizer
from tensorflow.contrib.ipu import ipu_infeed_queue
from tensorflow.contrib.ipu import ipu_outfeed_queue
from tensorflow.contrib.ipu import loops
from tensorflow.contrib.ipu import gradient_accumulation_optimizer


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


class ShardedAndReplicatedTest(test_util.TensorFlowTestCase):
  def testShardedAndReplicated(self):
    shape = [2]
    dataset = tu.create_single_increasing_dataset(3, shape)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name=next_feed_id(), replication_factor=2)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name=next_feed_id(), replication_factor=2)

    def body(v, x):
      with ipu.ops.ipu_shard(0):
        z = v + x
        y = x * x
      with ipu.ops.ipu_shard(1):
        z = popops_cross_replica_sum.cross_replica_sum(
            z) + popops_cross_replica_sum.cross_replica_sum(y)
        outfeed = outfeed_queue.enqueue(z)
      return (z, outfeed)

    def my_net():
      v = constant_op.constant(0.0, shape=shape, dtype=np.float32)
      r = loops.repeat(2, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = ipu_compiler.compile(my_net, inputs=[])

    outfed = outfeed_queue.dequeue()

    cfg = ipu.utils.create_ipu_config(
        profiling=True,
        max_cross_replica_sum_buffer_size=10000,
        max_inter_ipu_copies_buffer_size=10000)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 4)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(10, shape))
      outfed_result = sess.run(outfed)

      self.assertTrue(outfed_result.shape[0], 2)
      self.assertAllClose(outfed_result[0][0], outfed_result[0][1])
      self.assertAllClose(outfed_result[0][0], np.broadcast_to(2, shape))

      self.assertAllClose(outfed_result[1][0], outfed_result[1][1])
      self.assertAllClose(outfed_result[1][0], np.broadcast_to(10, shape))

  def testShardedAndReplicatedTraining(self):
    def my_graph(inp, lab):
      with ops.device("/device:IPU:0"):
        with ipu.ops.ipu_shard(0):
          x = layers.Conv2D(8, 3, padding='same', name="convA")(inp)

        with ipu.ops.ipu_shard(1):
          x = layers.Conv2D(8, 1, padding='same', name="convB")(x)
          x = math_ops.reduce_mean(x, axis=[1, 2])

          loss = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(lab))
          loss = math_ops.reduce_mean(loss)

        opt = ipu_optimizer.CrossReplicaOptimizer(
            ipu.sharded_optimizer.ShardedOptimizer(
                gradient_descent.GradientDescentOptimizer(0.000001)))
        train = opt.minimize(loss)

      return [loss, train]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
      lab = array_ops.placeholder(np.float32, [1, 8], name="labels")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [inp, lab])

    cfg = ipu.utils.create_ipu_config(
        profiling=True,
        max_cross_replica_sum_buffer_size=10000,
        max_inter_ipu_copies_buffer_size=10000)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 4)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {inp: np.ones([1, 32, 32, 4]), lab: np.ones([1, 8])}
      sess.run(out, fd)

      rep = sess.run(report)

      num_compiles = 0

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

      compile_report = ipu.utils.extract_compile_reports(rep)
      self.assertEqual(len(compile_report), 1)

      js = json.loads(compile_report[0][1])
      cs_list = js['computeSets']['names']

      # There are 8 inter-ipu communications
      n_inter_ipu_copies = 0
      for n in cs_list:
        if fnmatch.fnmatch(n, '*/GlobalPre/*'):
          n_inter_ipu_copies = n_inter_ipu_copies + 1

      self.assertEqual(n_inter_ipu_copies, 8)

  def testShardedAndReplicatedAndGradientAccumulateTraining(self):
    dataset = tu.create_dual_increasing_dataset(3)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name=next_feed_id(), replication_factor=2)

    def my_graph(loss, inp, lab):
      with ops.device("/device:IPU:0"):
        with ipu.ops.ipu_shard(0):
          x = layers.Conv2D(8, 3, padding='same', name="convA")(inp)

        with ipu.ops.ipu_shard(1):
          x = layers.Conv2D(8, 1, padding='same', name="convB")(x)
          x = math_ops.reduce_mean(x, axis=[1, 2])

          loss = nn.softmax_cross_entropy_with_logits_v2(
              logits=x, labels=array_ops.stop_gradient(lab))
          loss = math_ops.reduce_mean(loss)

        opt = gradient_accumulation_optimizer.CrossReplicaGradientAccumulationOptimizer(
            ipu.sharded_optimizer.ShardedOptimizer(
                gradient_descent.GradientDescentOptimizer(0.000001)), 10)
        train = opt.minimize(loss)

      return [loss, train]

    def my_net():
      v = 0.0
      r = loops.repeat(2, my_graph, [v], infeed_queue)
      return r

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_net, [])

    cfg = ipu.utils.create_ipu_config(
        profiling=True,
        max_cross_replica_sum_buffer_size=10000,
        max_inter_ipu_copies_buffer_size=10000)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 4)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      sess.run(out)

      rep = sess.run(report)

      num_compiles = 0

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

      compile_report = ipu.utils.extract_compile_reports(rep)
      self.assertEqual(len(compile_report), 1)

      js = json.loads(compile_report[0][1])
      cs_list = js['computeSets']['names']

      # There are 7 inter-ipu copies
      n_inter_ipu_copies = 0
      for n in cs_list:
        if fnmatch.fnmatch(n, '*/GlobalPre/*'):
          n_inter_ipu_copies = n_inter_ipu_copies + 1

      self.assertEqual(n_inter_ipu_copies, 7)


if __name__ == "__main__":
  googletest.main()
