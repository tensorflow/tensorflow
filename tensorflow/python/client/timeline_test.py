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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

from tensorflow.python.client import timeline


class TimelineTest(tf.test.TestCase):

  def _validateTrace(self, chrome_trace_format):
    # Check that the supplied string is valid JSON.
    trace = json.loads(chrome_trace_format)
    # It should have a top-level key containing events.
    self.assertTrue('traceEvents' in trace)
    # Every event in the list should have a 'ph' field.
    for event in trace['traceEvents']:
      self.assertTrue('ph' in event)

  def testSimpleTimeline(self):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with tf.device('/cpu:0'):
      with tf.Session() as sess:
        sess.run(
            tf.constant(1.0),
            options=run_options,
            run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    self._validateTrace(ctf)

  def testTimelineCpu(self):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with self.test_session(use_gpu=False) as sess:
      const1 = tf.constant(1.0, name='const1')
      const2 = tf.constant(2.0, name='const2')
      result = tf.add(const1, const2) + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/cpu:0' in devices)
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
    ctf = tl.generate_chrome_trace_format(show_memory=False,
                                          show_dataflow=False)
    self._validateTrace(ctf)

  def testTimelineGpu(self):
    if not tf.test.is_gpu_available(cuda_only=True):
      return

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with self.test_session(force_gpu=True) as sess:
      const1 = tf.constant(1.0, name='const1')
      const2 = tf.constant(2.0, name='const2')
      result = tf.add(const1, const2) + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/gpu:0' in devices)
    self.assertTrue('/gpu:0/stream:all' in devices)
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
    ctf = tl.generate_chrome_trace_format(show_memory=False,
                                          show_dataflow=False)
    self._validateTrace(ctf)

  def testTimelineWithRPCs(self):
    """Tests that Timeline can handle RPC tracing."""
    metadata = tf.RunMetadata()
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
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    config = tf.ConfigProto(device_count={'CPU': 3})

    with tf.Session(config=config) as sess:
      with tf.device('/cpu:0'):
        const1 = tf.constant(1.0, name='const1')
      with tf.device('/cpu:1'):
        const2 = tf.constant(2.0, name='const2')
      with tf.device('/cpu:2'):
        result = const1 + const2 + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)

    self.assertTrue(run_metadata.HasField('step_stats'))
    tl = timeline.Timeline(run_metadata.step_stats)
    step_analysis = tl.analyze_step_stats()
    ctf = step_analysis.chrome_trace.format_to_string()
    self._validateTrace(ctf)
    maximums = step_analysis.allocator_maximums
    self.assertTrue('cpu' in maximums)
    cpu_max = maximums['cpu']
    # At least const1 + const2, both float32s (4 bytes each)
    self.assertGreater(cpu_max.num_bytes, 8)
    self.assertGreater(cpu_max.timestamp, 0)
    self.assertTrue('const1' in cpu_max.tensors)
    self.assertTrue('const2' in cpu_max.tensors)

  def testManyCPUs(self):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    config = tf.ConfigProto(device_count={'CPU': 3})
    with tf.Session(config=config) as sess:
      with tf.device('/cpu:0'):
        const1 = tf.constant(1.0, name='const1')
      with tf.device('/cpu:1'):
        const2 = tf.constant(2.0, name='const2')
      with tf.device('/cpu:2'):
        result = const1 + const2 + const1 * const2
      sess.run(result, options=run_options, run_metadata=run_metadata)
    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/cpu:0' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/cpu:1' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/cpu:2' in devices)
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
    ctf = tl.generate_chrome_trace_format(show_memory=False,
                                          show_dataflow=False)
    self._validateTrace(ctf)


if __name__ == '__main__':
  tf.test.main()
