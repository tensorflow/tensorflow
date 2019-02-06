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
# ==============================================================================
"""Tests for call_trees module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class CallTreesTest(converter_testing.TestCase):

  def test_normal_function(self):

    def test_fn(f):
      return f() + 3

    with self.converted(test_fn, call_trees, {}) as result:
      self.assertEquals(
          result.test_fn(None),
          converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 3)
      self.assertListEqual(self.dynamic_calls, [()])

  def test_class_method(self):

    class TestClass(object):

      def test_method(self, a):
        return self.other_method(a) + 1

    tc = TestClass()
    with self.converted(TestClass.test_method, call_trees, {}) as result:
      self.assertEquals(converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 1,
                        result.test_method(tc, 1))
      self.assertListEqual(self.dynamic_calls, [(1,)])

  def test_object_method(self):

    class TestClass(object):

      def test_method(self, a):
        return self.other_method(a) + 1

    tc = TestClass()
    with self.converted(tc.test_method, call_trees, {}) as result:
      self.assertEquals(converter_testing.RESULT_OF_MOCK_CONVERTED_CALL + 1,
                        result.test_method(tc, 1))
      self.assertListEqual(self.dynamic_calls, [(1,)])


if __name__ == '__main__':
  test.main()
