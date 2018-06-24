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

from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import lists
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class ListTest(converter_test_base.TestCase):

  def test_empty_list(self):

    def test_fn():
      return []

    node = self.parse_and_analyze(test_fn, {})
    node = lists.transform(node, self.ctx)

    with self.compiled(node) as result:
      tl = result.test_fn()
      # Empty tensor lists cannot be evaluated or stacked.
      self.assertTrue(isinstance(tl, ops.Tensor))
      self.assertEqual(tl.dtype, dtypes.variant)

  def test_initialized_list(self):

    def test_fn():
      return [1, 2, 3]

    node = self.parse_and_analyze(test_fn, {})
    node = lists.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        tl = result.test_fn()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(sess.run(r), [1, 2, 3])

  def test_list_append(self):

    def test_fn():
      l = [1]
      l.append(2)
      l.append(3)
      return l

    node = self.parse_and_analyze(test_fn, {})
    node = lists.transform(node, self.ctx)

    with self.compiled(node) as result:
      with self.test_session() as sess:
        tl = result.test_fn()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(sess.run(r), [1, 2, 3])

  def test_list_pop(self):

    def test_fn():
      l = [1, 2, 3]
      utils.set_element_type(l, dtypes.int32, ())
      s = l.pop()
      return s, l

    node = self.parse_and_analyze(
        test_fn,
        {
            'utils': utils,
            'dtypes': dtypes
        },
        include_type_analysis=True,
    )
    node = lists.transform(node, self.ctx)

    with self.compiled(node) as result:
      result.utils = utils
      result.dtypes = dtypes
      with self.test_session() as sess:
        ts, tl = result.test_fn()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        self.assertAllEqual(sess.run(r), [1, 2])
        self.assertAllEqual(sess.run(ts), 3)

  def test_double_list_pop(self):

    def test_fn(l):
      s = l.pop().pop()
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = lists.transform(node, self.ctx)

    with self.compiled(node) as result:
      test_input = [1, 2, [1, 2, 3]]
      # TODO(mdan): Pass a list of lists of tensor when we fully support that.
      # For now, we just pass a regular Python list of lists just to verify that
      # the two pop calls are sequenced properly.
      self.assertAllEqual(result.test_fn(test_input), 3)

  def test_list_stack(self):

    tf = None  # Will be replaced with a mock.

    def test_fn():
      l = [1, 2, 3]
      utils.set_element_type(l, dtypes.int32)
      return tf.stack(l)

    node = self.parse_and_analyze(
        test_fn,
        {
            'utils': utils,
            'dtypes': dtypes
        },
        include_type_analysis=True,
    )
    node = lists.transform(node, self.ctx)

    with self.compiled(node, array_ops.stack, dtypes.int32) as result:
      result.utils = utils
      result.dtypes = dtypes
      with self.test_session() as sess:
        self.assertAllEqual(sess.run(result.test_fn()), [1, 2, 3])


if __name__ == '__main__':
  test.main()
