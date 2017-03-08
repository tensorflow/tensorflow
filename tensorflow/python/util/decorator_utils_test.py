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
"""decorator_utils tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils


def _test_function(unused_arg=0):
  pass


class GetQualifiedNameTest(tf.test.TestCase):

  def test_method(self):
    self.assertEqual(
        "GetQualifiedNameTest.test_method",
        decorator_utils.get_qualified_name(GetQualifiedNameTest.test_method))

  def test_function(self):
    self.assertEqual(
        "_test_function",
        decorator_utils.get_qualified_name(_test_function))


class AddNoticeToDocstringTest(tf.test.TestCase):

  def _check(self, doc, expected):
    self.assertEqual(
        decorator_utils.add_notice_to_docstring(
            doc=doc,
            instructions="Instructions",
            no_doc_str="Nothing here",
            suffix_str="(suffix)",
            notice=["Go away"]),
        expected)

  def test_regular(self):
    self._check("Brief\n\nDocstring",
                "Brief (suffix)\n\nGo away\nInstructions\n\nDocstring")

  def test_brief_only(self):
    self._check("Brief",
                "Brief (suffix)\n\nGo away\nInstructions")

  def test_no_docstring(self):
    self._check(None,
                "Nothing here\n\nGo away\nInstructions")
    self._check("",
                "Nothing here\n\nGo away\nInstructions")

  def test_no_empty_line(self):
    self._check("Brief\nDocstring",
                "Brief (suffix)\n\nGo away\nInstructions\n\nDocstring")


class ValidateCallableTest(tf.test.TestCase):

  def test_function(self):
    decorator_utils.validate_callable(_test_function, "test")

  def test_method(self):
    decorator_utils.validate_callable(self.test_method, "test")

  def test_callable(self):
    class TestClass(object):

      def __call__(self):
        pass
    decorator_utils.validate_callable(TestClass(), "test")

  def test_partial(self):
    partial = functools.partial(_test_function, unused_arg=7)
    decorator_utils.validate_callable(partial, "test")

  def test_fail_non_callable(self):
    x = 0
    self.assertRaises(ValueError, decorator_utils.validate_callable, x, "test")

if __name__ == "__main__":
  tf.test.main()
