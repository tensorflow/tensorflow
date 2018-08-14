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

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.python.platform import test


class OriginInfoTest(test.TestCase):

  def test_source_map(self):

    def test_fn(x):
      if x > 0:
        x += 1
      return x

    node, source = parser.parse_entity(test_fn)
    fn_node = node.body[0]
    origin_info.resolve(fn_node, source)

    # Insert a traced line.
    new_node = parser.parse_str('x = abs(x)').body[0]
    anno.copyanno(fn_node.body[0], new_node, anno.Basic.ORIGIN)
    fn_node.body.insert(0, new_node)

    # Insert an untraced line.
    fn_node.body.insert(0, parser.parse_str('x = 0').body[0])

    modified_source = compiler.ast_to_source(fn_node)

    source_map = origin_info.source_map(fn_node, modified_source,
                                        'test_filename', [0])

    loc = origin_info.LineLocation('test_filename', 1)
    origin = source_map[loc]
    self.assertEqual(origin.source_code_line, 'def test_fn(x):')
    self.assertEqual(origin.loc.lineno, 1)

    # The untraced line, inserted second.
    loc = origin_info.LineLocation('test_filename', 2)
    self.assertFalse(loc in source_map)

    # The traced line, inserted first.
    loc = origin_info.LineLocation('test_filename', 3)
    origin = source_map[loc]
    self.assertEqual(origin.source_code_line, '  if x > 0:')
    self.assertEqual(origin.loc.lineno, 2)

    loc = origin_info.LineLocation('test_filename', 4)
    origin = source_map[loc]
    self.assertEqual(origin.source_code_line, '  if x > 0:')
    self.assertEqual(origin.loc.lineno, 2)

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
