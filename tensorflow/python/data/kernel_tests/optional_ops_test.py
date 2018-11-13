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
"""Tests for the Optional data type wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class OptionalTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testFromValue(self):
    opt = optional_ops.Optional.from_value(constant_op.constant(37.0))
    self.assertEqual(dtypes.float32, opt.output_types)
    self.assertEqual([], opt.output_shapes)
    self.assertEqual(ops.Tensor, opt.output_classes)
    self.assertTrue(self.evaluate(opt.has_value()))
    self.assertEqual(37.0, self.evaluate(opt.get_value()))

  @test_util.run_in_graph_and_eager_modes
  def testFromStructuredValue(self):
    opt = optional_ops.Optional.from_value({
        "a": constant_op.constant(37.0),
        "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))
    })
    self.assertEqual({
        "a": dtypes.float32,
        "b": (dtypes.string, dtypes.string)
    }, opt.output_types)
    self.assertEqual({"a": [], "b": ([1], [])}, opt.output_shapes)
    self.assertEqual({
        "a": ops.Tensor,
        "b": (ops.Tensor, ops.Tensor)
    }, opt.output_classes)
    self.assertTrue(self.evaluate(opt.has_value()))
    self.assertEqual({
        "a": 37.0,
        "b": ([b"Foo"], b"Bar")
    }, self.evaluate(opt.get_value()))

  @test_util.run_in_graph_and_eager_modes
  def testFromSparseTensor(self):
    st_0 = sparse_tensor.SparseTensorValue(
        indices=np.array([[0]]),
        values=np.array([0], dtype=np.int64),
        dense_shape=np.array([1]))
    st_1 = sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 1]]),
        values=np.array([-1., 1.], dtype=np.float32),
        dense_shape=np.array([2, 2]))
    opt = optional_ops.Optional.from_value((st_0, st_1))
    self.assertEqual((dtypes.int64, dtypes.float32), opt.output_types)
    self.assertEqual(([1], [2, 2]), opt.output_shapes)
    self.assertEqual((sparse_tensor.SparseTensor, sparse_tensor.SparseTensor),
                     opt.output_classes)

  @test_util.run_in_graph_and_eager_modes
  def testFromNone(self):
    opt = optional_ops.Optional.none_from_structure(tensor_shape.scalar(),
                                                    dtypes.float32, ops.Tensor)
    self.assertEqual(dtypes.float32, opt.output_types)
    self.assertEqual([], opt.output_shapes)
    self.assertEqual(ops.Tensor, opt.output_classes)
    self.assertFalse(self.evaluate(opt.has_value()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(opt.get_value())

  def testStructureMismatchError(self):
    tuple_output_shapes = (tensor_shape.scalar(), tensor_shape.scalar())
    tuple_output_types = (dtypes.float32, dtypes.float32)
    tuple_output_classes = (ops.Tensor, ops.Tensor)

    dict_output_shapes = {
        "a": tensor_shape.scalar(),
        "b": tensor_shape.scalar()
    }
    dict_output_types = {"a": dtypes.float32, "b": dtypes.float32}
    dict_output_classes = {"a": ops.Tensor, "b": ops.Tensor}

    with self.assertRaises(TypeError):
      optional_ops.Optional.none_from_structure(
          tuple_output_shapes, tuple_output_types, dict_output_classes)

    with self.assertRaises(TypeError):
      optional_ops.Optional.none_from_structure(
          tuple_output_shapes, dict_output_types, tuple_output_classes)

    with self.assertRaises(TypeError):
      optional_ops.Optional.none_from_structure(
          dict_output_shapes, tuple_output_types, tuple_output_classes)

  @test_util.run_in_graph_and_eager_modes
  def testCopyToGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with ops.device("/cpu:0"):
      optional_with_value = optional_ops.Optional.from_value(
          (constant_op.constant(37.0), constant_op.constant("Foo"),
           constant_op.constant(42)))
      optional_none = optional_ops.Optional.none_from_structure(
          tensor_shape.scalar(), dtypes.float32, ops.Tensor)

    with ops.device("/gpu:0"):
      gpu_optional_with_value = optional_ops._OptionalImpl(
          array_ops.identity(optional_with_value._variant_tensor),
          optional_with_value.output_shapes, optional_with_value.output_types,
          optional_with_value.output_classes)
      gpu_optional_none = optional_ops._OptionalImpl(
          array_ops.identity(optional_none._variant_tensor),
          optional_none.output_shapes, optional_none.output_types,
          optional_none.output_classes)

      gpu_optional_with_value_has_value = gpu_optional_with_value.has_value()
      gpu_optional_with_value_values = gpu_optional_with_value.get_value()

      gpu_optional_none_has_value = gpu_optional_none.has_value()

    self.assertTrue(self.evaluate(gpu_optional_with_value_has_value))
    self.assertEqual((37.0, b"Foo", 42),
                     self.evaluate(gpu_optional_with_value_values))
    self.assertFalse(self.evaluate(gpu_optional_none_has_value))

  def testIteratorGetNextAsOptional(self):
    ds = dataset_ops.Dataset.range(3)
    iterator = ds.make_initializable_iterator()
    next_elem = iterator_ops.get_next_as_optional(iterator)
    self.assertTrue(isinstance(next_elem, optional_ops.Optional))
    self.assertEqual(ds.output_types, next_elem.output_types)
    self.assertEqual(ds.output_shapes, next_elem.output_shapes)
    self.assertEqual(ds.output_classes, next_elem.output_classes)
    elem_has_value_t = next_elem.has_value()
    elem_value_t = next_elem.get_value()
    with self.test_session() as sess:
      # Before initializing the iterator, evaluating the optional fails with
      # a FailedPreconditionError.
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(elem_has_value_t)
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(elem_value_t)

      # For each element of the dataset, assert that the optional evaluates to
      # the expected value.
      sess.run(iterator.initializer)
      for i in range(3):
        elem_has_value, elem_value = sess.run([elem_has_value_t, elem_value_t])
        self.assertTrue(elem_has_value)
        self.assertEqual(i, elem_value)

      # After exhausting the iterator, `next_elem.has_value()` will evaluate to
      # false, and attempting to get the value will fail.
      for _ in range(2):
        self.assertFalse(sess.run(elem_has_value_t))
        with self.assertRaises(errors.InvalidArgumentError):
          sess.run(elem_value_t)


if __name__ == "__main__":
  test.main()
