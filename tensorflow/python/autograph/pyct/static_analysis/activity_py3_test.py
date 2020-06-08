# python3
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
"""Tests for activity module, that only run in Python 3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct.static_analysis import activity_test
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.platform import test


NodeAnno = annos.NodeAnno


class ActivityAnalyzerTest(activity_test.ActivityAnalyzerTestBase):
  """Tests which can only run in Python 3."""

  def test_nonlocal_symbol(self):
    nonlocal_a = 3
    nonlocal_b = 13

    def test_fn(c):
      nonlocal nonlocal_a
      nonlocal nonlocal_b
      nonlocal_a = nonlocal_b + c

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node
    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(
        body_scope, ('nonlocal_a', 'nonlocal_b', 'c'), ('nonlocal_a',))
    nonlocal_a_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(nonlocal_a_scope, ('nonlocal_a',), ())

  def test_annotated_assign(self):
    b = int

    def test_fn(c):
      a: b = c
      return a

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node

    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('b', 'c', 'a'), ('a',))
    self.assertSymbolSetsAre(('b',), body_scope.annotations, 'annotations')

    ann_assign_scope = anno.getanno(fn_node.body[0], anno.Static.SCOPE)
    self.assertScopeIs(ann_assign_scope, ('b', 'c'), ('a',))
    self.assertSymbolSetsAre(
        ('b',), ann_assign_scope.annotations, 'annotations')

  def test_function_def_annotations(self):
    b = int
    c = int

    def test_fn(a: b) -> c:
      return a

    node, _ = self._parse_and_analyze(test_fn)
    fn_node = node

    fn_scope = anno.getanno(fn_node, anno.Static.SCOPE)
    self.assertScopeIs(fn_scope, ('b', 'c'), ('test_fn',))
    self.assertSymbolSetsAre(('b', 'c'), fn_scope.annotations, 'annotations')

    body_scope = anno.getanno(fn_node, NodeAnno.BODY_SCOPE)
    self.assertScopeIs(body_scope, ('a',), ())
    self.assertSymbolSetsAre((), body_scope.annotations, 'annotations')


if __name__ == '__main__':
  test.main()
