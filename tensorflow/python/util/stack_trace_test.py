# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for fast Python stack trace utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

from tensorflow.python import _stack_trace_binding_for_test
from tensorflow.python.platform import test


# Our stack tracing doesn't have source code line yet, so erase for now.
def erase_line(frame_summary):
  return [
      frame_summary.filename, frame_summary.lineno, frame_summary.name,
      '<source line unimplemented>'
  ]


class StackTraceTest(test.TestCase):

  def testStackTrace(self):
    our_stack = _stack_trace_binding_for_test.to_string()
    true_stack = traceback.extract_stack(limit=10)
    true_stack = [erase_line(fs) for fs in true_stack]
    true_stack[-1][1] -= 1  # true_stack capturing was one line below.
    true_stack = ''.join(traceback.format_list(true_stack))

    self.assertEqual(our_stack, true_stack)


if __name__ == '__main__':
  test.main()
