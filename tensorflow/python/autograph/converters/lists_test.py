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
"""Tests for lists module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


tf = None  # Will be replaced by a mock.


class ListTest(converter_testing.TestCase):

  def test_empty_list(self):

    def test_fn():
      return []

    with self.converted(test_fn, lists, {}) as result:
      tl = result.test_fn()
      # Empty tensor lists cannot be evaluated or stacked.
      self.assertTrue(isinstance(tl, ops.Tensor))
      self.assertEqual(tl.dtype, dtypes.variant)

  def test_initialized_list(self):

    def test_fn():
      return [1, 2, 3]

    with self.converted(test_fn, lists, {}) as result:
      self.assertAllEqual(result.test_fn(), [1, 2, 3])

  def test_list_append(self):

    def test_fn():
      l = special_functions.tensor_list([1])
      l.append(2)
      l.append(3)
      return l

    ns = {'special_functions': special_functions}
    with self.converted(test_fn, lists, ns) as result:
      with self.cached_session() as sess:
        tl = result.test_fn()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(self.evaluate(r), [1, 2, 3])

  def test_list_pop(self):

    def test_fn():
      l = special_functions.tensor_list([1, 2, 3])
      s = l.pop()
      return s, l

    ns = {'special_functions': special_functions}
    node, ctx = self.prepare(test_fn, ns)
    def_, = anno.getanno(node.body[0].targets[0],
                         anno.Static.ORIG_DEFINITIONS)
    def_.directives[directives.set_element_type] = {
        'dtype': parser.parse_expression('tf.int32'),
        'shape': parser.parse_expression('()'),
    }
    node = lists.transform(node, ctx)

    with self.compiled(node, ns, dtypes.int32) as result:
      with self.cached_session() as sess:
        ts, tl = result.test_fn()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(self.evaluate(r), [1, 2])
        self.assertAllEqual(self.evaluate(ts), 3)

  def test_double_list_pop(self):

    def test_fn(l):
      s = l.pop().pop()
      return s

    with self.converted(test_fn, lists, {}) as result:
      test_input = [1, 2, [1, 2, 3]]
      # TODO(mdan): Pass a list of lists of tensor when we fully support that.
      # For now, we just pass a regular Python list of lists just to verify that
      # the two pop calls are sequenced properly.
      self.assertAllEqual(result.test_fn(test_input), 3)

  def test_list_stack(self):

    def test_fn():
      l = [1, 2, 3]
      return tf.stack(l)

    node, ctx = self.prepare(test_fn, {})
    def_, = anno.getanno(node.body[0].targets[0],
                         anno.Static.ORIG_DEFINITIONS)
    def_.directives[directives.set_element_type] = {
        'dtype': parser.parse_expression('tf.int32')
    }
    node = lists.transform(node, ctx)

    with self.compiled(node, {}, array_ops.stack, dtypes.int32) as result:
      with self.cached_session() as sess:
        self.assertAllEqual(self.evaluate(result.test_fn()), [1, 2, 3])

  # TODO(mdan): Add a test with tf.stack with axis kwarg.


if __name__ == '__main__':
  test.main()
