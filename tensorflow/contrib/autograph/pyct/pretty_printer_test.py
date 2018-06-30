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
"""Tests for pretty_printer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

from tensorflow.contrib.autograph.pyct import pretty_printer
from tensorflow.python.platform import test


class PrettyPrinterTest(test.TestCase):

  def test_format(self):
    node = ast.FunctionDef(
        name='f',
        args=ast.arguments(
            args=[ast.Name(id='a', ctx=ast.Param())],
            vararg=None,
            kwarg=None,
            defaults=[]),
        body=[
            ast.Return(
                ast.BinOp(
                    op=ast.Add(),
                    left=ast.Name(id='a', ctx=ast.Load()),
                    right=ast.Num(1)))
        ],
        decorator_list=[],
        returns=None)
    # Just checking for functionality, the color control characters make it
    # difficult to inspect the result.
    self.assertIsNotNone(pretty_printer.fmt(node))


if __name__ == '__main__':
  test.main()
