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
"""Tests for data_structures module."""

from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class ListTest(test.TestCase):

  def test_new_list_empty(self):
    l = data_structures.new_list()
    # Can't evaluate an empty list.
    # TODO(mdan): sess.run should allow tf.variant maybe?
    self.assertTrue(isinstance(l, ops.Tensor))

  def test_new_list_tensor(self):
    l = data_structures.new_list([3, 4, 5])
    self.assertAllEqual(l, [3, 4, 5])

  def test_tf_tensor_list_new(self):
    l = data_structures.tf_tensor_list_new([3, 4, 5])
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  def test_tf_tensor_list_new_empty(self):
    l = data_structures.tf_tensor_list_new([],
                                           element_dtype=dtypes.int32,
                                           element_shape=())
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [])

  def test_tf_tensor_list_new_from_tensor(self):
    l = data_structures.tf_tensor_list_new(constant_op.constant([3, 4, 5]))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  @test_util.run_deprecated_v1
  def test_tf_tensor_list_new_illegal_input(self):
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4.0])
    # TODO(mdan): It might make more sense to type cast in this case.
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4], element_dtype=dtypes.float32)
    # Tensor lists do support heterogeneous lists.
    self.assertIsNot(data_structures.tf_tensor_list_new([3, [4, 5]]), None)
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4], element_shape=(2,))
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new(
          constant_op.constant([1, 2, 3]), element_shape=[1])

  def test_tf_tensor_array_new(self):
    l = data_structures.tf_tensor_array_new([3, 4, 5])
    t = l.stack()
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  def test_tf_tensor_array_new_illegal_input(self):
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4.0])
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4], element_dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, [4, 5]])
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4], element_shape=(2,))
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([], element_shape=(2,))
    # TAs can infer the shape.
    self.assertIsNot(
        data_structures.tf_tensor_array_new([], element_dtype=dtypes.float32),
        None)

  def test_append_tensor_list(self):
    l = data_structures.new_list()
    x = constant_op.constant([1, 2, 3])
    l = data_structures.list_append(l, x)

    t = list_ops.tensor_list_stack(l, element_dtype=x.dtype)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [[1, 2, 3]])

  @test_util.run_deprecated_v1
  def test_append_tensorarray(self):
    l = tensor_array_ops.TensorArray(dtypes.int32, size=0, dynamic_size=True)
    l1 = data_structures.list_append(l, 1)
    l2 = data_structures.list_append(l1, 2)

    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(l1.stack()), [1])
      self.assertAllEqual(self.evaluate(l2.stack()), [1, 2])

  def test_append_python(self):
    l = []
    self.assertAllEqual(data_structures.list_append(l, 1), [1])
    self.assertAllEqual(data_structures.list_append(l, 2), [1, 2])

  def test_pop_tensor_list(self):
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)

    opts = data_structures.ListPopOpts(
        element_dtype=initial_list.dtype,
        element_shape=(2,))

    with self.assertRaises(NotImplementedError):
      data_structures.list_pop(l, 0, opts)

    with self.cached_session() as sess:
      l, x = data_structures.list_pop(l, None, opts)
      self.assertAllEqual(self.evaluate(x), [3, 4])

      t = list_ops.tensor_list_stack(l, element_dtype=initial_list.dtype)
      self.assertAllEqual(self.evaluate(t), [[1, 2]])

  def test_pop_python(self):
    l = [1, 2, 3]
    opts = data_structures.ListPopOpts(element_dtype=None, element_shape=())
    self.assertAllEqual(data_structures.list_pop(l, None, opts), ([1, 2], 3))
    self.assertAllEqual(data_structures.list_pop(l, None, opts), ([1], 2))

  def test_stack_tensor_list(self):
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)

    opts = data_structures.ListStackOpts(
        element_dtype=initial_list.dtype, original_call=None)

    with self.cached_session() as sess:
      t = data_structures.list_stack(l, opts)
      self.assertAllEqual(self.evaluate(t), self.evaluate(initial_list))

  @test_util.run_deprecated_v1
  def test_stack_tensor_list_empty(self):
    l = list_ops.empty_tensor_list(
        element_shape=None, element_dtype=dtypes.variant)

    opts = data_structures.ListStackOpts(
        element_dtype=dtypes.int32, original_call=None)

    # TODO(mdan): Allow stacking empty lists if the dtype and shape are known.
    with self.assertRaises(ValueError):
      data_structures.list_stack(l, opts)

  def test_stack_fallback(self):

    def dummy_function(l):
      # Lazy person's mock: just transform the argument in a way in which we
      # can check that this function was indeed called.
      return [x * 2 for x in l]

    opts = data_structures.ListStackOpts(
        element_dtype=None, original_call=dummy_function)

    self.assertAllEqual(data_structures.list_stack([1, 2], opts), [2, 4])


if __name__ == '__main__':
  test.main()
