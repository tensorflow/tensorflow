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
"""Tests for anno module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.python.platform import test


# TODO(mdan): Consider strong types instead of primitives.


class AnnoTest(test.TestCase):

  def test_basic(self):
    node = ast.Name()

    self.assertFalse(anno.hasanno(node, 'foo'))
    with self.assertRaises(AttributeError):
      anno.getanno(node, 'foo')

    anno.setanno(node, 'foo', 3)
    self.assertTrue(anno.hasanno(node, 'foo'))
    self.assertEqual(anno.getanno(node, 'foo'), 3)
    self.assertEqual(anno.getanno(node, 'bar', default=7), 7)

    anno.delanno(node, 'foo')
    self.assertFalse(anno.hasanno(node, 'foo'))
    with self.assertRaises(AttributeError):
      anno.getanno(node, 'foo')
    self.assertIsNone(anno.getanno(node, 'foo', default=None))

  def test_copyanno(self):
    node_1 = ast.Name()
    anno.setanno(node_1, 'foo', 3)

    node_2 = ast.Name()
    anno.copyanno(node_1, node_2, 'foo')
    anno.copyanno(node_1, node_2, 'bar')

    self.assertTrue(anno.hasanno(node_2, 'foo'))
    self.assertFalse(anno.hasanno(node_2, 'bar'))


if __name__ == '__main__':
  test.main()
