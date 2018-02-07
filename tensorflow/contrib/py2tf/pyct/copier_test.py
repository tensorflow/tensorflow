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
"""Tests for copier module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

from tensorflow.contrib.py2tf.pyct import copier
from tensorflow.python.platform import test


class CopierTest(test.TestCase):

  def test_copy_clean(self):
    ret = ast.Return(
        ast.BinOp(
            op=ast.Add(),
            left=ast.Name(id='a', ctx=ast.Load()),
            right=ast.Num(1)))
    setattr(ret, '__foo', 'bar')
    node = ast.FunctionDef(
        name='f',
        args=ast.arguments(
            args=[ast.Name(id='a', ctx=ast.Param())],
            vararg=None,
            kwarg=None,
            defaults=[]),
        body=[ret],
        decorator_list=[],
        returns=None)
    new_node = copier.copy_clean(node)
    self.assertFalse(node is new_node)
    self.assertFalse(ret is new_node.body[0])
    self.assertFalse(hasattr(new_node.body[0], '__foo'))


if __name__ == '__main__':
  test.main()
