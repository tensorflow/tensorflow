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
"""Tests for tensorflow.python.client.Timeline."""

import json

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TimelineTest(test.TestCase):

  def _validateTrace(self, chrome_trace_format):
    # Check that the supplied string is valid JSON.
    trace = json.loads(chrome_trace_format)
    # It should have a top-level key containing events.
    self.assertTrue('traceEvents' in trace)
    # Every event in the list should have a 'ph' field.
    for event in trace['traceEvents']:
      self.assertTrue('ph' in event)

  def testSimpleTimeline(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with ops.device('/cpu:0'):
      with session.Session() as sess:
        sess.run(constant_op.constant(1.0),
                 options=run_options,
                 run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)

  @test_util.deprecated_graph_mode_only
  def testTimelineCpu(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with self.session(use_gpu=False) as sess:
      const1 = constant_op.constant(1.0, name='const1')
      const2 = constant_op.constant(2.0, name='const2')
      result = math_ops.add(const1, const2) + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in devices)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_dataflow=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(
        show_memory=False, show_dataflow=False)
    self._validateTrace(ctf)

  @test_util.deprecated_graph_mode_only
  def testTimelineGpu(self):
    if not test.is_gpu_available(cuda_only=True):
      return

    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    with self.session(force_gpu=True) as sess:
      const1 = constant_op.constant(1.0, name='const1')
      const2 = constant_op.constant(2.0, name='const2')
      result = math_ops.add(const1, const2) + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:0' in devices)
    self.assertIn('/device:GPU:0/stream:all', devices)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_dataflow=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(
        show_memory=False, show_dataflow=False)
    self._validateTrace(ctf)

  def testTimelineWithRPCs(self):
    """Tests that Timeline can handle RPC tracing."""
    metadata = config_pb2.RunMetadata()
    step_stats = metadata.step_stats
    dev_stats = step_stats.dev_stats.add()
    dev_stats.device = '/job:worker/replica:0/task:0/cpu:0'
    node_stats = dev_stats.node_stats.add()
    node_stats.node_name = 'RecvTensor'
    node_stats.all_start_micros = 12345
    node_stats.op_end_rel_micros = 42
    node_stats.timeline_label = ('[1024B] edge_160_conv2/biases/read from '
                                 '/job:ps/replica:0/task:3/cpu:0 to '
                                 '/job:worker/replica:0/task:0/cpu:0')
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)

  def testAnalysisAndAllocations(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    config = config_pb2.ConfigProto(device_count={'CPU': 3})

    with session.Session(config=config) as sess:
      with ops.device('/cpu:0'):
        num1 = variables.Variable(1.0, name='num1')
      with ops.device('/cpu:1'):
        num2 = variables.Variable(2.0, name='num2')
      with ops.device('/cpu:2'):
        result = num1 + num2 + num1 * num2
      self.evaluate(variables.global_variables_initializer())
      sess.run(result, options=run_options, run_metadata=run_metadata)

    self.assertTrue(run_metadata.HasField('step_stats'))
    tl = timeline.Timeline(run_metadata.step_stats)
    step_analysis = tl.analyze_step_stats()
    ctf = step_analysis.chrome_trace.format_to_string()
    self._validateTrace(ctf)
    maximums = step_analysis.allocator_maximums
    cpuname = 'mklcpu' if test_util.IsMklEnabled() else 'cpu'
    self.assertTrue(cpuname in maximums)
    cpu_max = maximums[
        'cuda_host_bfc'] if 'cuda_host_bfc' in maximums else maximums[cpuname]
    # At least num1 + num2, both float32s (4 bytes each)
    self.assertGreaterEqual(cpu_max.num_bytes, 8)
    self.assertGreater(cpu_max.timestamp, 0)

  def testManyCPUs(self):
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    config = config_pb2.ConfigProto(device_count={'CPU': 3})
    with session.Session(config=config) as sess:
      with ops.device('/cpu:0'):
        num1 = variables.Variable(1.0, name='num1')
      with ops.device('/cpu:1'):
        num2 = variables.Variable(2.0, name='num2')
      with ops.device('/cpu:2'):
        result = num1 + num2 + num1 * num2
      self.evaluate(variables.global_variables_initializer())
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:1' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:2' in devices)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_dataflow=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=False)
    self._validateTrace(ctf)
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(
        show_memory=False, show_dataflow=False)
    self._validateTrace(ctf)


if __name__ == '__main__':
  test.main()
