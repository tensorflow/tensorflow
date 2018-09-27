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
    node = activity.resolve(node, entity_info)
    graphs = cfg.build(node)
    liveness.resolve(node, entity_info, graphs)
    return node

  def assertHasLiveOut(self, node, expected):
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    live_out_str = set(str(v) for v in live_out)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(live_out_str, set(expected))

  def test_stacked_if(self):

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

  def test_stacked_if_else(self):

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

  def test_for_basic(self):

    def test_fn(x, a):
      for i in range(a):
        x += i
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'x')

  def test_attributes(self):

    def test_fn(x, a):
      if a > 0:
        x.y = 0
      return x.y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], ('x.y', 'x'))

  def test_nested_functions(self):

    def test_fn(a, b):
      if b:
        a = []

      def foo():
        return a

      foo()

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasLiveOut(fn_body[0], 'a')

  def test_nested_functions_isolation(self):

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


if __name__ == '__main__':
  test.main()
