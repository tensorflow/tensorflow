# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for liveness module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.platform import test


class LivenessTest(test.TestCase):

  def _parse_and_analyze(self, test_fn):
    node, source = parser.parse_entity(test_fn)
    entity_info = transformer.EntityInfo(
        source_code=source,
        source_file=None,
        namespace={},
        arg_values=None,
        arg_types=None,
        owner_type=None)
    node = qual_names.resolve(node)
    ctx = transformer.Context(entity_info)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    liveness.resolve(node, ctx, graphs)
    return node

  def assertHasLiveOut(self, node, expected):
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    live_out_strs = set(str(v) for v in live_out)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(live_out_strs, set(expected))

  def assertHasLiveIn(self, node, expected):
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_in_strs = set(str(v) for v in live_in)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(live_in_strs, set(expected))

  def test_live_out_stacked_if(self):

    def test_fn(x, a):
      if a > 0:
        x = 0
      if a > 1:
        x = 1
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], ('a', 'x'))
    self.assertHasLiveOut(fn_body[1], 'x')

  def test_live_out_stacked_if_else(self):

    def test_fn(x, a):
      if a > 0:
        x = 0
      if a > 1:
        x = 1
      else:
        x = 2
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'a')
    self.assertHasLiveOut(fn_body[1], 'x')

  def test_live_out_for_basic(self):

    def test_fn(x, a):
      for i in range(a):
        x += i
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'x')

  def test_live_out_attributes(self):

    def test_fn(x, a):
      if a > 0:
        x.y = 0
      return x.y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], ('x.y', 'x'))

  def test_live_out_nested_functions(self):

    def test_fn(a, b):
      if b:
        a = []

      def foo():
        return a

      foo()

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'a')

  def test_live_out_nested_functions_isolation(self):

    def test_fn(b):
      if b:
        a = 0  # pylint:disable=unused-variable

      def child():
        max(a)  # pylint:disable=used-before-assignment
        a = 1
        return a

      child()

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'max')

  def test_live_out_deletion(self):

    def test_fn(x, y, a):
      for _ in a:
        if x:
          del y
        else:
          y = 0

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], ())

  def test_live_in_stacked_if(self):

    def test_fn(x, a, b, c):
      if a > 0:
        x = b
      if c > 1:
        x = 0
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'x'))
    self.assertHasLiveIn(fn_body[1], ('c', 'x'))

  def test_live_in_stacked_if_else(self):

    def test_fn(x, a, b, c, d):
      if a > 1:
        x = b
      else:
        x = c
      if d > 0:
        x = 0
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'd'))
    self.assertHasLiveIn(fn_body[1], ('d', 'x'))

  def test_live_in_for_basic(self):

    def test_fn(x, y, a):
      for i in a:
        x = i
        y += x
        z = 0
      return y, z

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

  def test_live_in_for_nested(self):

    def test_fn(x, y, a):
      for i in a:
        for j in i:
          x = i
          y += x
          z = j
      return y, z

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

  def test_live_in_deletion(self):

    def test_fn(x, y, a):
      for _ in a:
        if x:
          del y
        else:
          y = 0

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveIn(fn_body[0], ('a', 'x', 'y'))

  def test_live_in_generator_comprehension(self):

    def test_fn(y):
      if all(x for x in y):
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    if six.PY2:
      self.assertHasLiveIn(fn_body[0], ('all', 'x', 'y'))
    else:
      self.assertHasLiveIn(fn_body[0], ('all', 'y'))

  def test_live_in_list_comprehension(self):

    def test_fn(y):
      if [x for x in y]:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    if six.PY2:
      self.assertHasLiveIn(fn_body[0], ('x', 'y'))
    else:
      self.assertHasLiveIn(fn_body[0], ('y',))

  def test_live_in_set_comprehension(self):

    def test_fn(y):
      if {x for x in y}:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    if six.PY2:
      self.assertHasLiveIn(fn_body[0], ('x', 'y'))
    else:
      self.assertHasLiveIn(fn_body[0], ('y',))

  def test_live_in_dict_comprehension(self):

    def test_fn(y):
      if {k: v for k, v in y}:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    if six.PY2:
      self.assertHasLiveIn(fn_body[0], ('k', 'v', 'y'))
    else:
      self.assertHasLiveIn(fn_body[0], ('y',))


if __name__ == '__main__':
  test.main()
