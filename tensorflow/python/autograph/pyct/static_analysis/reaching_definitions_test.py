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
"""Tests for reaching_definitions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.platform import test


class DefinitionInfoTest(test.TestCase):

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
    node = reaching_definitions.resolve(node, entity_info, graphs,
                                        reaching_definitions.Definition)
    return node

  def assertHasDefs(self, node, num):
    defs = anno.getanno(node, anno.Static.DEFINITIONS)
    self.assertEqual(len(defs), num)
    for r in defs:
      self.assertIsInstance(r, reaching_definitions.Definition)

  def assertHasDefinedIn(self, node, expected):
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    defined_in_str = set(str(v) for v in defined_in)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(defined_in_str, set(expected))

  def assertSameDef(self, first, second):
    self.assertHasDefs(first, 1)
    self.assertHasDefs(second, 1)
    self.assertIs(
        anno.getanno(first, anno.Static.DEFINITIONS)[0],
        anno.getanno(second, anno.Static.DEFINITIONS)[0])

  def assertNotSameDef(self, first, second):
    self.assertHasDefs(first, 1)
    self.assertHasDefs(second, 1)
    self.assertIsNot(
        anno.getanno(first, anno.Static.DEFINITIONS)[0],
        anno.getanno(second, anno.Static.DEFINITIONS)[0])

  def test_conditional(self):

    def test_fn(a, b):
      a = []
      if b:
        a = []
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))

  def test_while(self):

    def test_fn(a):
      max(a)
      while True:
        a = a
        a = a
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasDefs(fn_body[0].value.args[0], 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].body[1].targets[0], 1)
    self.assertHasDefs(fn_body[1].body[1].value, 1)
    # The loop does have an invariant test, but the CFG doesn't know that.
    self.assertHasDefs(fn_body[1].body[0].value, 2)
    self.assertHasDefs(fn_body[2].value, 2)

  def test_while_else(self):

    def test_fn(x, i):
      y = 0
      while x:
        x += i
        if i:
          break
      else:
        y = 1
      return x, y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 2)
    self.assertHasDefs(fn_body[1].body[0].target, 1)
    self.assertHasDefs(fn_body[1].body[1].test, 1)
    self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value.elts[0], 2)
    self.assertHasDefs(fn_body[2].value.elts[1], 2)

  def test_for_else(self):

    def test_fn(x, i):
      y = 0
      for i in x:
        x += i
        if i:
          break
        else:
          continue
      else:
        y = 1
      return x, y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].target, 1)
    self.assertHasDefs(fn_body[1].body[0].target, 1)
    self.assertHasDefs(fn_body[1].body[1].test, 1)
    self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value.elts[0], 2)
    self.assertHasDefs(fn_body[2].value.elts[1], 2)

  def test_nested_functions(self):

    def test_fn(a, b):
      a = []
      if b:
        a = []

        def foo():
          return a

        foo()

      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body
    def_of_a_in_if = fn_body[1].body[0].targets[0]

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(def_of_a_in_if, 1)
    self.assertHasDefs(fn_body[2].value, 2)

    inner_fn_body = fn_body[1].body[1].body
    self.assertSameDef(inner_fn_body[0].value, def_of_a_in_if)

  def test_nested_functions_isolation(self):

    def test_fn(a):
      a = 0

      def child():
        a = 1
        return a

      child()
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    parent_return = fn_body[3]
    child_return = fn_body[1].body[1]
    # The assignment `a = 1` makes `a` local to `child`.
    self.assertNotSameDef(parent_return.value, child_return.value)

  def test_function_call_in_with(self):

    def foo(_):
      pass

    def test_fn(a):
      with foo(a):
        return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    self.assertHasDefs(fn_body[0].items[0].context_expr.func, 0)
    self.assertHasDefs(fn_body[0].items[0].context_expr.args[0], 1)

  def test_mutation_subscript(self):

    def test_fn(a):
      l = []
      l[0] = a
      return l

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    creation = fn_body[0].targets[0]
    mutation = fn_body[1].targets[0].value
    use = fn_body[2].value
    self.assertSameDef(creation, mutation)
    self.assertSameDef(creation, use)

  def test_deletion_partial(self):

    def test_fn(a):
      a = 0
      if a:
        del a
      else:
        a = 1
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    first_def = fn_body[0].targets[0]
    second_def = fn_body[1].orelse[0].targets[0]
    use = fn_body[2].value
    self.assertNotSameDef(use, first_def)
    self.assertSameDef(use, second_def)

  def test_deletion_total(self):

    def test_fn(a):
      if a:
        a = 0
      else:
        a = 1
      del a
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    use = fn_body[2].value
    self.assertHasDefs(use, 0)

  def test_replacement(self):

    def foo(a):
      return a

    def test_fn(a):
      a = foo(a)
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body[0].body

    param = node.body[0].args.args[0]
    source = fn_body[0].value.args[0]
    target = fn_body[0].targets[0]
    retval = fn_body[1].value
    self.assertSameDef(param, source)
    self.assertNotSameDef(source, target)
    self.assertSameDef(target, retval)


if __name__ == '__main__':
  test.main()
