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

import six

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.platform import test


global_a = 7
global_b = 17


class ReachingDefinitionsAnalyzerTestBase(test.TestCase):

  def _parse_and_analyze(self, test_fn):
    # TODO(mdan): Use a custom FunctionTransformer here.
    node, source = parser.parse_entity(test_fn, future_features=())
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file=None,
        future_features=(),
        namespace={})
    node = qual_names.resolve(node)
    namer = naming.Namer({})
    ctx = transformer.Context(entity_info, namer, None)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    node = reaching_definitions.resolve(node, ctx, graphs,
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


class ReachingDefinitionsAnalyzerTest(ReachingDefinitionsAnalyzerTestBase):

  def test_conditional(self):

    def test_fn(a, b):
      a = []
      if b:
        a = []
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))

  def test_try_in_conditional(self):

    def test_fn(a, b):  # pylint:disable=unused-argument
      a = []
      if b:
        try:
          pass
        except:  # pylint:disable=bare-except
          pass
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))

  def test_conditional_in_try_in_conditional(self):

    def test_fn(a, b):
      a = []
      if b:
        try:
          if b:
            a = []
        except TestException:  # pylint:disable=undefined-variable,unused-variable
          pass
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
    # Note: `TestException` and `e` are not tracked.
    self.assertHasDefinedIn(fn_body[1].body[0].body[0], ('a', 'b'))

  def test_conditional_in_except_in_conditional(self):

    def test_fn(a, b):
      a = []
      if b:
        try:
          pass
        except TestException as e:  # pylint:disable=undefined-variable,unused-variable
          if b:
            a = []
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
    # Note: `TestException` and `e` are not tracked.
    self.assertHasDefinedIn(fn_body[1].body[0].handlers[0].body[0], ('a', 'b'))

  def test_while(self):

    def test_fn(a):
      max(a)
      while True:
        a = a
        a = a
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

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
    fn_body = node.body

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
    fn_body = node.body

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
    fn_body = node.body
    def_of_a_in_if = fn_body[1].body[0].targets[0]

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(def_of_a_in_if, 1)
    self.assertHasDefs(fn_body[2].value, 2)

    inner_fn_body = fn_body[1].body[1].body
    def_of_a_in_foo = inner_fn_body[0].value
    # Even though `a` is visible in the inner functio above, the late binding
    # makes it impossible to assume that the same value will be visible at
    # call time.
    self.assertHasDefs(def_of_a_in_foo, 0)

  def test_nested_functions_isolation(self):

    def test_fn(a):
      a = 0

      def child():
        a = 1
        return a

      child()
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

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
    fn_body = node.body

    self.assertHasDefs(fn_body[0].items[0].context_expr.func, 0)
    self.assertHasDefs(fn_body[0].items[0].context_expr.args[0], 1)

  def test_mutation_subscript(self):

    def test_fn(a):
      l = []
      l[0] = a
      return l

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

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
    fn_body = node.body

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
    fn_body = node.body

    use = fn_body[2].value
    self.assertHasDefs(use, 0)

  def test_replacement(self):

    def foo(a):
      return a

    def test_fn(a):
      a = foo(a)
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    param = node.args.args[0]
    source = fn_body[0].value.args[0]
    target = fn_body[0].targets[0]
    retval = fn_body[1].value
    self.assertSameDef(param, source)
    self.assertNotSameDef(source, target)
    self.assertSameDef(target, retval)

  def test_comprehension_leaking(self):

    def test_fn(a):
      _ = [x for x in a]
      return x  # pylint:disable=undefined-loop-variable

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    listcomp_target = fn_body[0].value.generators[0].target
    retval = fn_body[1].value

    # Python2 leaks list comprehension symbols. Python3 doesn't.
    # For details, see:
    # https://stackoverflow.com/questions/4198906/list-comprehension-rebinds-names-even-after-scope-of-comprehension-is-this-righ
    if six.PY2:
      self.assertSameDef(retval, listcomp_target)
    else:
      self.assertHasDefs(retval, 0)

  def test_function_definition(self):

    def test_fn():
      def a():
        pass
      if a:  # pylint:disable=using-constant-test
        a = None
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a',))

  def test_global(self):

    def test_fn():
      global global_a
      global global_b
      if global_a:
        global_b = []
      return global_a, global_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('global_a', 'global_b'))


if __name__ == '__main__':
  test.main()
