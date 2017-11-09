# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tpu_function helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import tpu_function

from tensorflow.python.platform import test


class FunctionArgCheckTest(test.TestCase):

  def testSimple(self):
    """Tests that arg checker works for functions with no varargs or defaults.
    """

    def func(x, y, z):
      return x + y + z

    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, None))
    self.assertEqual("exactly 3 arguments",
                     tpu_function.check_function_argument_count(func, 2, None))
    queue = tpu_feed.InfeedQueue(2)
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 1, queue))
    self.assertEqual("exactly 3 arguments",
                     tpu_function.check_function_argument_count(func, 2, queue))

  def testDefaultArgs(self):
    """Tests that arg checker works for a function with no varargs."""

    def func(x, y, z=17):
      return x + y + z

    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, None))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 1, None))
    self.assertEqual("at most 3 arguments",
                     tpu_function.check_function_argument_count(func, 4, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 1, queue))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 0, queue))
    self.assertEqual("at most 3 arguments",
                     tpu_function.check_function_argument_count(func, 4, queue))

  def testVarArgs(self):
    """Tests that arg checker works for a function with varargs."""

    def func(x, y, *z):
      return x + y + len(z)

    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 4, None))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 1, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, queue))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 0, queue))

  def testVarArgsAndDefaults(self):
    """Tests that arg checker works for a function with varargs and defaults."""

    def func(x, y, z=17, *q):
      return x + y + z + len(q)

    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 4, None))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 5, None))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 1, None))
    queue = tpu_feed.InfeedQueue(1)
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 1, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 2, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 3, queue))
    self.assertEqual(None,
                     tpu_function.check_function_argument_count(func, 4, queue))
    self.assertEqual("at least 2 arguments",
                     tpu_function.check_function_argument_count(func, 0, queue))


if __name__ == "__main__":
  test.main()
