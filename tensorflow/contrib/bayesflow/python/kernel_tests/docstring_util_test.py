# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
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
"""Tests for docstring utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.bayesflow.python.ops import docstring_util
from tensorflow.python.platform import test


class DocstringUtil(test.TestCase):

  def _testFunction(self):
    doc_args = """x: Input to return as output.
  y: Baz."""
    @docstring_util.expand_docstring(args=doc_args)
    def foo(x):
      # pylint: disable=g-doc-args
      """Hello world.

      Args:
        @{args}

      Returns:
        x.
      """
      # pylint: enable=g-doc-args
      return x

    true_docstring = """Hello world.

    Args:
      x: Input to return as output.
      y: Baz.

    Returns:
      x.
    """
    self.assertEqual(foo.__doc__, true_docstring)

  def _testClassInit(self):
    doc_args = """x: Input to return as output.
  y: Baz."""

    class Foo(object):

      @docstring_util.expand_docstring(args=doc_args)
      def __init__(self, x, y):
        # pylint: disable=g-doc-args
        """Hello world.

        Args:
          @{args}

        Bar.
        """
        # pylint: enable=g-doc-args
        pass

    true_docstring = """Hello world.

    Args:
      x: Input to return as output.
      y: Baz.

    Bar.
    """
    self.assertEqual(Foo.__doc__, true_docstring)


if __name__ == "__main__":
  test.main()
