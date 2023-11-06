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
# ==============================================================================
"""Tests for functions used to extract and analyze stacks."""

from tensorflow.python.platform import test
from tensorflow.python.util import tf_stack


class TFStackTest(test.TestCase):

  def testFrameSummaryEquality(self):
    frames1 = tf_stack.extract_stack()
    frames2 = tf_stack.extract_stack()

    self.assertNotEqual(frames1[0], frames1[1])
    self.assertEqual(frames1[0], frames1[0])
    self.assertEqual(frames1[0], frames2[0])

  def testFrameSummaryEqualityAndHash(self):
    # Both defined on the same line to produce identical stacks.
    frame1, frame2 = tf_stack.extract_stack(), tf_stack.extract_stack()
    self.assertEqual(len(frame1), len(frame2))
    for f1, f2 in zip(frame1, frame2):
      self.assertEqual(f1, f2)
      self.assertEqual(hash(f1), hash(f1))
      self.assertEqual(hash(f1), hash(f2))
    self.assertEqual(frame1, frame2)
    self.assertEqual(hash(tuple(frame1)), hash(tuple(frame2)))

  def testLastUserFrame(self):
    trace = tf_stack.extract_stack()
    frame = trace.last_user_frame()
    self.assertRegex(repr(frame), 'testLastUserFrame')

  def testGetUserFrames(self):

    def func():
      trace = tf_stack.extract_stack()  # COMMENT
      frames = list(trace.get_user_frames())
      return frames

    frames = func()  # CALLSITE

    self.assertRegex(repr(frames[-1]), 'func')
    self.assertRegex(repr(frames[-2]), 'testGetUserFrames')

  def testGetItem(self):
    def func(n):
      if n == 0:
        return tf_stack.extract_stack()  # COMMENT
      else:
        return func(n - 1)

    trace = func(5)
    self.assertIn('func', repr(trace[-1]))

    with self.assertRaises(IndexError):
      _ = trace[-len(trace) - 1]

    with self.assertRaises(IndexError):
      _ = trace[len(trace)]

  def testSourceMap(self):
    source_map = tf_stack._tf_stack.PyBindSourceMap()

    def func(n):
      if n == 0:
        return tf_stack._tf_stack.extract_stack(
            source_map, tf_stack._tf_stack.PyBindFileSet()
        )
      else:
        return func(n - 1)

    trace = func(5)
    source_map.update_to((
        (
            (trace[0].filename, trace[0].lineno),
            ("filename", 42, "function_name"),
        ),
    ))
    trace = list(func(5))
    self.assertEqual(
        str(trace[0]), 'File "filename", line 42, in function_name'
    )

  def testStackTraceBuilder(self):
    stack1 = tf_stack.extract_stack()
    stack2 = tf_stack.extract_stack()
    stack3 = tf_stack.extract_stack()

    builder = tf_stack.GraphDebugInfoBuilder()
    builder.AccumulateStackTrace('func1', 'node1', stack1)
    builder.AccumulateStackTrace('func2', 'node2', stack2)
    builder.AccumulateStackTrace('func3', 'node3', stack3)
    debug_info = builder.Build()

    trace_map = tf_stack.LoadTracesFromDebugInfo(debug_info)
    self.assertSameElements(
        trace_map.keys(), ['node1@func1', 'node2@func2', 'node3@func3']
    )

    for trace in trace_map.values():
      self.assertRegex(repr(trace), 'tf_stack_test.py', trace)


if __name__ == "__main__":
  test.main()
