# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""test the RunMetadata proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# pylint: disable=g-bad-import-order
# XXX: this depends on pywrap_tensorflow and must come later
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib
SIZE = 1300


def _extract_node(run_meta, node_name):
  ret = defaultdict(list)
  for dev_stat in run_meta.step_stats.dev_stats:
    dev = dev_stat.device
    for node_stat in dev_stat.node_stats:
      if node_stat.node_name == node_name:
        ret[dev].append(node_stat)
  return ret


def _run_model():
  x = random_ops.random_normal(shape=[1, SIZE])
  w = random_ops.random_normal(shape=[SIZE, 2 * SIZE])
  y = math_ops.matmul(x, w)

  with session.Session() as sess:
    run_metadata = config_pb2.RunMetadata()
    opts = model_analyzer.PRINT_ALL_TIMING_MEMORY
    opts['min_micros'] = 0
    opts['min_bytes'] = 0
    _ = sess.run(y,
                 options=config_pb2.RunOptions(
                     trace_level=config_pb2.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata)
    tfprof_node = model_analyzer.profile(
        sess.graph,
        run_meta=run_metadata,
        options=opts)

    return tfprof_node, run_metadata


def _run_loop_model():
  with session.Session() as sess:
    x = lib.BuildFullModel()

    sess.run(variables.global_variables_initializer())
    run_meta = config_pb2.RunMetadata()
    _ = sess.run(x,
                 options=config_pb2.RunOptions(
                     trace_level=config_pb2.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)

    tfprof_node = model_analyzer.profile(
        sess.graph, run_meta,
        options=model_analyzer.PRINT_ALL_TIMING_MEMORY)
    return tfprof_node, run_meta


class RunMetadataTest(test.TestCase):

  def testGPU(self):
    if not test.is_gpu_available(cuda_only=True):
      return

    ops.reset_default_graph()
    with ops.device('/gpu:0'):
      tfprof_node, run_meta = _run_model()
      self.assertEqual(tfprof_node.children[0].name, 'MatMul')
      self.assertGreater(tfprof_node.children[0].exec_micros, 10)

    ret = _extract_node(run_meta, 'MatMul')
    self.assertEqual(len(ret), 1)
    self.assertTrue('/job:localhost/replica:0/task:0/gpu:0' in ret)

    ret = _extract_node(run_meta, 'MatMul:MatMul')
    self.assertEqual(len(ret), 2)
    has_all_stream = False
    for k, _ in six.iteritems(ret):
      self.assertTrue('gpu:0/stream' in k)
      if 'gpu:0/stream:all' in k:
        has_all_stream = True
    self.assertTrue(has_all_stream)

  def testCPU(self):
    ops.reset_default_graph()
    with ops.device('/cpu:0'):
      tfprof_node, run_meta = _run_model()
      self.assertEqual(tfprof_node.children[0].name, 'MatMul')
      self.assertGreater(tfprof_node.children[0].exec_micros, 0)

    ret = _extract_node(run_meta, 'MatMul')
    self.assertEqual(len(ret), 1)
    self.assertTrue('/job:localhost/replica:0/task:0/cpu:0' in ret)

    ret = _extract_node(run_meta, 'MatMul:MatMul')
    self.assertEqual(len(ret), 0)

  def testLoopCPU(self):
    ops.reset_default_graph()
    with ops.device('/cpu:0'):
      tfprof_node, run_meta = _run_loop_model()
      # The while-loop caused a node to appear 4 times in scheduling.
      ret = _extract_node(run_meta,
                          'rnn/while/rnn/basic_rnn_cell/basic_rnn_cell/MatMul')
      self.assertEqual(len(ret['/job:localhost/replica:0/task:0/cpu:0']), 4)

      total_cpu_execs = 0
      for node in ret['/job:localhost/replica:0/task:0/cpu:0']:
        total_cpu_execs += node.op_end_rel_micros

      mm_node = lib.SearchTFProfNode(
          tfprof_node,
          'rnn/while/rnn/basic_rnn_cell/basic_rnn_cell/MatMul')

      self.assertEqual(mm_node.run_count, 4)
      self.assertEqual(mm_node.cpu_exec_micros, total_cpu_execs)
      self.assertEqual(mm_node.exec_micros, total_cpu_execs)

  # pylint: disable=pointless-string-statement
  """
  TODO(xpan): This test is flaky because RunMetadata returned from TensorFlow
  is random. Still being investigated.
  def testLoopGPU(self):
    if not test.is_gpu_available():
      return

    ops.reset_default_graph()
    with ops.device('/gpu:0'):
      tfprof_node, run_meta = _run_loop_model()
      # The while-loop caused a node to appear 4 times in scheduling.
      ret = _extract_node(run_meta,
                          'rnn/while/rnn/basic_rnn_cell/basic_rnn_cell/MatMul')
      self.assertEqual(len(ret['/job:localhost/replica:0/task:0/gpu:0']), 4)

      total_cpu_execs = 0
      for node in ret['/job:localhost/replica:0/task:0/gpu:0']:
        total_cpu_execs += node.op_end_rel_micros

      ret = _extract_node(
          run_meta,
          'rnn/while/rnn/basic_rnn_cell/basic_rnn_cell/MatMul:MatMul')
      self.assertGreaterEqual(len(ret['/gpu:0/stream:all']), 4)

      total_accelerator_execs = 0
      for node in ret['/gpu:0/stream:all']:
        total_accelerator_execs += node.op_end_rel_micros

      mm_node = lib.SearchTFProfNode(
          tfprof_node,
          'rnn/while/rnn/basic_rnn_cell/basic_rnn_cell/MatMul')

      self.assertEqual(mm_node.run_count, 4)
      self.assertEqual(mm_node.accelerator_exec_micros, total_accelerator_execs)
      self.assertEqual(mm_node.cpu_exec_micros, total_cpu_execs)
      self.assertEqual(mm_node.exec_micros,
                       total_cpu_execs + total_accelerator_execs)
  """


if __name__ == '__main__':
  test.main()
