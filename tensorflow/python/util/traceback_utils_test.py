# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for traceback_utils."""

import traceback

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import traceback_utils


class TracebackUtilsTest(test.TestCase):

  def assert_trace_line_count(self, fn, count, filtering_enabled=True):
    trace_line_count = -1
    if filtering_enabled:
      traceback_utils.enable_traceback_filtering()
    else:
      traceback_utils.disable_traceback_filtering()
    self.assertEqual(
        traceback_utils.is_traceback_filtering_enabled(), filtering_enabled)
    try:
      fn()
    except Exception as e:  # pylint: disable=broad-except
      # We must count lines rather than frames because autograph transforms
      # stack frames into a single large string
      trace = '\n'.join(traceback.format_tb(e.__traceback__))
      trace_line_count = len(trace.split('\n'))

    self.assertGreater(trace_line_count, 0)

    if filtering_enabled:
      self.assertLess(trace_line_count, count)
    else:
      self.assertGreater(trace_line_count, count)

  def test_eager_add(self):

    def fn():
      x = array_ops.zeros((2, 3))
      y = array_ops.zeros((2, 4))
      _ = x + y

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_tfn_add(self):
    @def_function.function
    def fn():
      x = array_ops.zeros((2, 3))
      y = array_ops.zeros((2, 4))
      return x + y

    self.assert_trace_line_count(fn, count=10, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_tfn_div(self):
    @def_function.function
    def wrapped_fn(x):
      return x / 0.

    def fn():
      wrapped_fn(0.5)

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

  def test_eager_argmax(self):
    def fn():
      _ = math_ops.argmax([0, 1], axis=2)

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)

  def test_tfn_argmax(self):
    @def_function.function
    def wrapped_fn(x):
      return math_ops.argmax(x, axis=2)

    def fn():
      wrapped_fn([0, 1])

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=25, filtering_enabled=False)

  def test_variable_constructor(self):
    def fn():
      _ = variables.Variable()

    self.assert_trace_line_count(fn, count=15, filtering_enabled=True)
    self.assert_trace_line_count(fn, count=30, filtering_enabled=False)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
