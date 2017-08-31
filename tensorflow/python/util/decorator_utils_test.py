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

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils


def _test_function(unused_arg=0):
  pass


class GetQualifiedNameTest(test.TestCase):

  def test_method(self):
    self.assertEqual(
        "GetQualifiedNameTest.test_method",
        decorator_utils.get_qualified_name(GetQualifiedNameTest.test_method))

  def test_function(self):
    self.assertEqual("_test_function",
                     decorator_utils.get_qualified_name(_test_function))


class AddNoticeToDocstringTest(test.TestCase):

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
    expected = ("Brief (suffix)\n\nGo away\nInstructions\n\nDocstring\n\n"
                "Args:\n  arg1: desc")
    # No indent for main docstring
    self._check("Brief\n\nDocstring\n\nArgs:\n  arg1: desc", expected)
    # 2 space indent for main docstring, blank lines not indented
    self._check("Brief\n\n  Docstring\n\n  Args:\n    arg1: desc", expected)
    # 2 space indent for main docstring, blank lines indented as well.
    self._check("Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc", expected)
    # No indent for main docstring, first line blank.
    self._check("\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc",
                expected)
    # 2 space indent, first line blank.
    self._check("\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc",
                expected)

  def test_brief_only(self):
    expected = "Brief (suffix)\n\nGo away\nInstructions"
    self._check("Brief", expected)
    self._check("Brief\n", expected)
    self._check("Brief\n  ", expected)
    self._check("\nBrief\n  ", expected)
    self._check("\n  Brief\n  ", expected)

  def test_no_docstring(self):
    expected = "Nothing here\n\nGo away\nInstructions"
    self._check(None, expected)
    self._check("", expected)

  def test_no_empty_line(self):
    expected = "Brief (suffix)\n\nGo away\nInstructions\n\nDocstring"
    # No second line indent
    self._check("Brief\nDocstring", expected)
    # 2 space second line indent
    self._check("Brief\n  Docstring", expected)
    # No second line indent, first line blank
    self._check("\nBrief\nDocstring", expected)
    # 2 space second line indent, first line blank
    self._check("\n  Brief\n  Docstring", expected)


class ValidateCallableTest(test.TestCase):

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
  test.main()
