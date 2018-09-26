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
"""Tests for origin_info module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.platform import test


class OriginInfoTest(test.TestCase):

  def test_create_source_map(self):

    def test_fn(x):
      return x + 1

    node, _ = parser.parse_entity(test_fn)
    fake_origin = origin_info.OriginInfo(
        loc=origin_info.Location('fake_filename', 3, 7),
        function_name='fake_function_name',
        source_code_line='fake source line',
        comment=None)
    fn_node = node.body[0]
    anno.setanno(fn_node.body[0], anno.Basic.ORIGIN, fake_origin)
    converted_code = compiler.ast_to_source(fn_node)

    source_map = origin_info.create_source_map(
        fn_node, converted_code, 'test_filename', [0])

    loc = origin_info.LineLocation('test_filename', 2)
    self.assertIn(loc, source_map)
    self.assertIs(source_map[loc], fake_origin)

  def test_source_map_no_origin(self):

    def test_fn(x):
      return x + 1

    node, _ = parser.parse_entity(test_fn)
    fn_node = node.body[0]
    converted_code = compiler.ast_to_source(fn_node)

    source_map = origin_info.create_source_map(
        fn_node, converted_code, 'test_filename', [0])

    self.assertEqual(len(source_map), 0)

  def test_resolve(self):

    def test_fn(x):
      """Docstring."""
      return x  # comment

    node, source = parser.parse_entity(test_fn)
    fn_node = node.body[0]

    origin_info.resolve(fn_node, source)

    origin = anno.getanno(fn_node, anno.Basic.ORIGIN)
    self.assertEqual(origin.loc.lineno, 1)
    self.assertEqual(origin.loc.col_offset, 0)
    self.assertEqual(origin.source_code_line, 'def test_fn(x):')
    self.assertIsNone(origin.comment)

    origin = anno.getanno(fn_node.body[0], anno.Basic.ORIGIN)
    self.assertEqual(origin.loc.lineno, 2)
    self.assertEqual(origin.loc.col_offset, 2)
    self.assertEqual(origin.source_code_line, '  """Docstring."""')
    self.assertIsNone(origin.comment)

    origin = anno.getanno(fn_node.body[1], anno.Basic.ORIGIN)
    self.assertEqual(origin.loc.lineno, 3)
    self.assertEqual(origin.loc.col_offset, 2)
    self.assertEqual(origin.source_code_line, '  return x  # comment')
    self.assertEqual(origin.comment, 'comment')


if __name__ == '__main__':
  test.main()
