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
"""Tests for tensorflow/python/util/stack_trace.h."""

import traceback

from tensorflow.python.platform import test
from tensorflow.python.util import _stack_trace_test_lib


class StackTraceTest(test.TestCase):

  def testStackTraceMatchesPy(self):
    cc_stack_trace = _stack_trace_test_lib.GetStackTraceString()
    py_stack_trace = ''.join(traceback.format_stack())

    # Should be same except at the end where the stace trace generation calls
    # are made.
    self.assertEqual(
        cc_stack_trace.split('\n')[:-4],
        py_stack_trace.split('\n')[:-3])


if __name__ == '__main__':
  test.main()
