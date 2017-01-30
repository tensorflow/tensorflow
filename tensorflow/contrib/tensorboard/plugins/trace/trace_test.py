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
"""Tests for tensorflow.contrib.tensorboard.plugins.trace package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from google.protobuf import json_format

from tensorflow.contrib.tensorboard.plugins import trace
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class TraceTest(test.TestCase):

  def setUp(self):
    self._temp_dir = tempfile.mkdtemp()
    self._temp_trace_json = self._temp_dir + 'trace.json'

  def tearDown(self):
    gfile.DeleteRecursively(self._temp_dir)

  def testEmptyGraph(self):
    trace_info = self._store_and_read_trace_info()
    self.assertEqual(len(trace_info.ops), 0)

  def testHasSourceCodeOfThisFile(self):
    constant_op.constant(0)
    trace_info = self._store_and_read_trace_info()

    self.assertTrue(trace_info.files)
    for file_info in trace_info.files:
      if file_info.file_path.endswith('trace_test.py'):
        return
    self.fail('trace_test file not found in the trace info json')

  def testHasTheConstantOp(self):
    constant_op.constant(0)
    trace_info = self._store_and_read_trace_info()

    self.assertTrue(trace_info.ops)

    for op in trace_info.ops:
      if op.op_type == 'Const':
        return
    self.fail('Could not find operation of type `Const` in the graph')

  def testMultilineStatements(self):
    source = """def test():
      a(4,
        3,
        1)

      b(3, 4, 5)

      c((4, 3),
        (),
      )
    """
    line2start = trace.find_multiline_statements(source)

    self.assertEqual(line2start[3], 1)
    self.assertEqual(line2start[9], 7)
    self.assertEqual(len(line2start), 2)

  def _store_and_read_trace_info(self):
    trace.store_trace_info(self._temp_trace_json)
    trace_info = trace.TraceInfo()

    with gfile.Open(self._temp_trace_json) as f:
      text = f.read().decode('utf-8')
    json_format.Parse(text, trace_info)

    return trace_info


if __name__ == '__main__':
  test.main()
