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
"""Tests for ast_util module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

from tensorflow.contrib.py2tf.pyct import ast_util
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct import qual_names
from tensorflow.python.platform import test


class AstUtilTest(test.TestCase):

  def test_rename_symbols(self):
    node = ast.Tuple([
        ast.Name('a', ast.Load()),
        ast.Name('b', ast.Load()),
        ast.Attribute(ast.Name('b', None), 'c', ast.Store()),
        ast.Attribute(
            ast.Attribute(ast.Name('b', None), 'c', ast.Load()), 'd', None)
    ], None)
    node = qual_names.resolve(node)
    node = ast_util.rename_symbols(
        node, {
            qual_names.QN('a'):
                qual_names.QN('renamed_a'),
            qual_names.QN(qual_names.QN('b'), attr='c'):
                qual_names.QN('renamed_b_c'),
        })

    self.assertEqual(node.elts[0].id, 'renamed_a')
    self.assertTrue(isinstance(node.elts[0].ctx, ast.Load))
    self.assertEqual(node.elts[1].id, 'b')
    self.assertEqual(node.elts[2].id, 'renamed_b_c')
    self.assertTrue(isinstance(node.elts[2].ctx, ast.Store))
    self.assertEqual(node.elts[3].value.id, 'renamed_b_c')
    self.assertTrue(isinstance(node.elts[3].value.ctx, ast.Load))

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
    new_node = ast_util.copy_clean(node)
    self.assertFalse(node is new_node)
    self.assertFalse(ret is new_node.body[0])
    self.assertFalse(hasattr(new_node.body[0], '__foo'))

  def test_keywords_to_dict(self):
    keywords = parser.parse_expression('f(a=b, c=1, d=\'e\')').keywords
    d = ast_util.keywords_to_dict(keywords)
    # Make sure we generate a usable dict node by attaching it to a variable and
    # compiling everything.
    output = parser.parse_str('b = 3')
    output.body += (ast.Assign([ast.Name(id='d', ctx=ast.Store())], d),)
    result, _ = compiler.ast_to_object(output)
    self.assertDictEqual(result.d, {'a': 3, 'c': 1, 'd': 'e'})
    print(d)


if __name__ == '__main__':
  test.main()
