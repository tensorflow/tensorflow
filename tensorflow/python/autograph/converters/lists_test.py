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

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class ListTest(converter_testing.TestCase):

  def test_empty_list(self):

    def f():
      return []

    tr = self.transform(f, lists)

    tl = tr()
    # Empty tensor lists cannot be evaluated or stacked.
    self.assertIsInstance(tl, ops.Tensor)
    self.assertEqual(tl.dtype, dtypes.variant)

  def test_initialized_list(self):

    def f():
      return [1, 2, 3]

    tr = self.transform(f, lists)

    self.assertAllEqual(tr(), [1, 2, 3])

  def test_list_append(self):

    def f():
      l = special_functions.tensor_list([1])
      l.append(2)
      l.append(3)
      return l

    tr = self.transform(f, lists)

    tl = tr()
    r = list_ops.tensor_list_stack(tl, dtypes.int32)
    self.assertAllEqual(self.evaluate(r), [1, 2, 3])

  def test_list_pop(self):

    def f():
      l = special_functions.tensor_list([1, 2, 3])
      directives.set_element_type(l, dtype=dtypes.int32, shape=())
      s = l.pop()
      return s, l

    tr = self.transform(f, (directives_converter, lists))

    ts, tl = tr()
    r = list_ops.tensor_list_stack(tl, dtypes.int32)
    self.assertAllEqual(self.evaluate(r), [1, 2])
    self.assertAllEqual(self.evaluate(ts), 3)

  def test_double_list_pop(self):

    def f(l):
      s = l.pop().pop()
      return s

    tr = self.transform(f, lists)

    test_input = [1, 2, [1, 2, 3]]
    # TODO(mdan): Pass a list of lists of tensor when we fully support that.
    # For now, we just pass a regular Python list of lists just to verify that
    # the two pop calls are sequenced properly.
    self.assertAllEqual(tr(test_input), 3)

  def test_list_stack(self):

    def f():
      l = [1, 2, 3]
      return array_ops.stack(l)

    tr = self.transform(f, lists)

    self.assertAllEqual(self.evaluate(tr()), [1, 2, 3])


if __name__ == '__main__':
  test.main()
