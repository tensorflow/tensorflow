# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for pprof_profiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

from proto import profile_pb2
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.profiler import pprof_profiler


class PprofProfilerTest(test.TestCase):

  def testDataEmpty(self):
    output_dir = test.get_temp_dir()
    run_metadata = config_pb2.RunMetadata()
    graph = test.mock.MagicMock()
    graph.get_operations.return_value = []

    profiles = pprof_profiler.get_profiles(graph, run_metadata)
    self.assertEqual(0, len(profiles))
    profile_files = pprof_profiler.profile(
        graph, run_metadata, output_dir)
    self.assertEqual(0, len(profile_files))

  def testRunMetadataEmpty(self):
    output_dir = test.get_temp_dir()
    run_metadata = config_pb2.RunMetadata()
    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = 'Add/123'
    op1.traceback = [('a/b/file1', 10, 'some_var')]
    op1.type = 'add'
    graph.get_operations.return_value = [op1]

    profiles = pprof_profiler.get_profiles(graph, run_metadata)
    self.assertEqual(0, len(profiles))
    profile_files = pprof_profiler.profile(
        graph, run_metadata, output_dir)
    self.assertEqual(0, len(profile_files))

  def testValidProfile(self):
    output_dir = test.get_temp_dir()
    run_metadata = config_pb2.RunMetadata()

    node1 = step_stats_pb2.NodeExecStats(
        node_name='Add/123',
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = 'deviceA'
    device1.node_stats.extend([node1])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = 'Add/123'
    op1.traceback = [
        ('a/b/file1', 10, 'apply_op', 'abc'), ('a/c/file2', 12, 'my_op', 'def')]
    op1.type = 'add'
    graph.get_operations.return_value = [op1]

    expected_proto = """sample_type {
  type: 5
  unit: 5
}
sample_type {
  type: 6
  unit: 7
}
sample_type {
  type: 8
  unit: 7
}
sample {
  value: 1
  value: 4
  value: 2
  label {
    key: 1
    str: 2
  }
  label {
    key: 3
    str: 4
  }
}
string_table: ""
string_table: "node_name"
string_table: "Add/123"
string_table: "op_type"
string_table: "add"
string_table: "count"
string_table: "all_time"
string_table: "nanoseconds"
string_table: "op_time"
string_table: "Device 1 of 1: deviceA"
comment: 9
"""
    # Test with protos
    profiles = pprof_profiler.get_profiles(graph, run_metadata)
    self.assertEqual(1, len(profiles))
    self.assertTrue('deviceA' in profiles)
    self.assertEqual(expected_proto, str(profiles['deviceA']))
    # Test with files
    profile_files = pprof_profiler.profile(
        graph, run_metadata, output_dir)
    self.assertEqual(1, len(profile_files))
    with gzip.open(profile_files[0]) as profile_file:
      profile_contents = profile_file.read()
      profile = profile_pb2.Profile()
      profile.ParseFromString(profile_contents)
      self.assertEqual(expected_proto, str(profile))

  @test_util.run_v1_only('b/120545219')
  def testProfileWithWhileLoop(self):
    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()

    num_iters = 5
    with self.cached_session() as sess:
      i = constant_op.constant(0)
      c = lambda i: math_ops.less(i, num_iters)
      b = lambda i: math_ops.add(i, 1)
      r = control_flow_ops.while_loop(c, b, [i])
      sess.run(r, options=options, run_metadata=run_metadata)
      profiles = pprof_profiler.get_profiles(sess.graph, run_metadata)
      self.assertEqual(1, len(profiles))
      profile = next(iter(profiles.values()))
      add_samples = []  # Samples for the while/Add node
      for sample in profile.sample:
        if profile.string_table[sample.label[0].str] == 'while/Add':
          add_samples.append(sample)
      # Values for same nodes are aggregated.
      self.assertEqual(1, len(add_samples))
      # Value of "count" should be equal to number of iterations.
      self.assertEqual(num_iters, add_samples[0].value[0])


if __name__ == '__main__':
  test.main()
