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
"""Tests for call_trees module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.autograph.converters import call_trees
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CallTreesTest(converter_testing.TestCase):

  def test_basic(self):

    def test_fn_1(_):
      raise ValueError('This should not be called in the compiled version.')

    def other_test_fn_1(a):
      return a + 1

    def test_fn_2(a):
      return test_fn_1(a) + 1

    ns = {'test_fn_1': test_fn_1}
    node, ctx = self.prepare(test_fn_2, ns)
    node = call_trees.transform(node, ctx)

    with self.compiled(node, ns) as result:
      new_name, _ = ctx.namer.compiled_function_name(('test_fn_1',))
      setattr(result, new_name, other_test_fn_1)
      self.assertEquals(result.test_fn_2(1), 3)

  def test_dynamic_function(self):

    def test_fn_1():
      raise ValueError('This should be masked by the mock in self.compiled.')

    def test_fn_2(f):
      return f() + 3

    with self.converted(test_fn_2, call_trees, {}) as result:
      # 10 = 7 (from the mock) + 3 (from test_fn_2)
      self.assertEquals(10, result.test_fn_2(test_fn_1))

  def test_basic_method(self):

    class TestClass(object):

      def test_fn_1(self, a):
        return a + 1

      def test_fn_2(self, a):
        return self.test_fn_1(a) + 1

    ns = {'TestClass': TestClass}
    node, ctx = self.prepare(
        TestClass.test_fn_2,
        ns,
        namer=converter_testing.FakeNoRenameNamer(),
        arg_types={'self': (TestClass.__name__, TestClass)})
    node = call_trees.transform(node, ctx)

    with self.compiled(node, ns) as result:
      tc = TestClass()
      self.assertEquals(3, result.test_fn_2(tc, 1))

  def test_py_func_no_retval(self):

    def test_fn(a):
      setattr(a, 'foo', 'bar')

    with self.converted(test_fn, call_trees, {'setattr': setattr}) as result:
      with self.cached_session() as sess:

        class Dummy(object):
          pass

        a = Dummy()
        result.test_fn(a)
        py_func_op, = sess.graph.get_operations()
        self.assertFalse(hasattr(a, 'foo'))
        sess.run(py_func_op)
        self.assertEquals('bar', a.foo)

  def test_py_func_known_function(self):

    def test_fn():
      return np.random.binomial(2, 0.5)

    with self.converted(test_fn, call_trees, {'np': np},
                        dtypes.int64) as result:
      with self.cached_session() as sess:
        self.assertTrue(isinstance(result.test_fn(), ops.Tensor))
        self.assertIn(sess.run(result.test_fn()), (0, 1, 2))

  def test_uncompiled_modules(self):

    def test_fn(a):
      a = math_ops.multiply(a, constant_op.constant(2))
      a = math_ops.add(a, constant_op.constant(1))
      return a

    ns = {'math_ops': math_ops, 'constant_op': constant_op}
    node, ctx = self.prepare(
        test_fn,
        ns,
        arg_types=set(((math_ops.__name__,), (constant_op.__name__,))))
    node = call_trees.transform(node, ctx)

    with self.compiled(node, ns) as result:
      with self.cached_session() as sess:
        result_tensor = result.test_fn(constant_op.constant(1))
        self.assertEquals(sess.run(result_tensor), 3)


if __name__ == '__main__':
  test.main()
