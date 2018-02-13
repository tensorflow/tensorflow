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
"""Tests for qual_names module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct import qual_names
from tensorflow.python.platform import test


class QNTest(test.TestCase):

  def test_basic(self):
    a = qual_names.QN('a')
    self.assertEqual(a.qn, ('a',))
    self.assertEqual(str(a), 'a')
    self.assertEqual(a.ssf(), 'a')
    self.assertEqual(a.ast().id, 'a')
    self.assertFalse(a.is_composite())
    with self.assertRaises(ValueError):
      _ = a.parent

    a_b = qual_names.QN(a, 'b')
    self.assertEqual(a_b.qn, ('a', 'b'))
    self.assertEqual(str(a_b), 'a.b')
    self.assertEqual(a_b.ssf(), 'a_b')
    self.assertEqual(a_b.ast().value.id, 'a')
    self.assertEqual(a_b.ast().attr, 'b')
    self.assertTrue(a_b.is_composite())
    self.assertEqual(a_b.parent.qn, ('a',))

    a2 = qual_names.QN(a)
    self.assertEqual(a2.qn, ('a',))
    with self.assertRaises(ValueError):
      _ = a.parent

    a_b2 = qual_names.QN(a_b)
    self.assertEqual(a_b2.qn, ('a', 'b'))
    self.assertEqual(a_b2.parent.qn, ('a',))

    self.assertTrue(a2 == a)
    self.assertFalse(a2 is a)

    self.assertTrue(a_b.parent == a)
    self.assertTrue(a_b2.parent == a)

    self.assertTrue(a_b2 == a_b)
    self.assertFalse(a_b2 is a_b)
    self.assertFalse(a_b2 == a)

    with self.assertRaises(ValueError):
      qual_names.QN('a', 'b')

  def test_hashable(self):
    d = {qual_names.QN('a'): 'a', qual_names.QN('b'): 'b'}

    self.assertEqual(d[qual_names.QN('a')], 'a')
    self.assertEqual(d[qual_names.QN('b')], 'b')
    self.assertTrue(qual_names.QN('c') not in d)


class QNResolverTest(test.TestCase):

  def assertQNStringIs(self, node, qn_str):
    self.assertEqual(str(anno.getanno(node, anno.Basic.QN)), qn_str)

  def test_resolve(self):
    samples = """
      a
      a.b
      (c, d.e)
      [f, (g.h.i)]
      j(k, l)
    """
    nodes = qual_names.resolve(parser.parse_str(textwrap.dedent(samples)))
    nodes = tuple(n.value for n in nodes.body)

    self.assertQNStringIs(nodes[0], 'a')
    self.assertQNStringIs(nodes[1], 'a.b')
    self.assertQNStringIs(nodes[2].elts[0], 'c')
    self.assertQNStringIs(nodes[2].elts[1], 'd.e')
    self.assertQNStringIs(nodes[3].elts[0], 'f')
    self.assertQNStringIs(nodes[3].elts[1], 'g.h.i')
    self.assertQNStringIs(nodes[4].func, 'j')
    self.assertQNStringIs(nodes[4].args[0], 'k')
    self.assertQNStringIs(nodes[4].args[1], 'l')


if __name__ == '__main__':
  test.main()
