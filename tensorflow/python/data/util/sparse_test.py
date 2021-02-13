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
"""Tests for utilities working with arbitrarily nested structures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class SparseTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject("Case_0", {
                  "classes": (),
                  "expected": False
              }),
              combinations.NamedObject("Case_1", {
                  "classes": (ops.Tensor),
                  "expected": False
              }),
              combinations.NamedObject("Case_2", {
                  "classes": (((ops.Tensor))),
                  "expected": False
              }),
              combinations.NamedObject("Case_3", {
                  "classes": (ops.Tensor, ops.Tensor),
                  "expected": False
              }),
              combinations.NamedObject("Case_4", {
                  "classes": (ops.Tensor, sparse_tensor.SparseTensor),
                  "expected": True
              }),
              combinations.NamedObject("Case_5", {
                  "classes": (sparse_tensor.SparseTensor,
                              sparse_tensor.SparseTensor),
                  "expected": True
              }),
              combinations.NamedObject("Case_6", {
                  "classes": (((sparse_tensor.SparseTensor))),
                  "expected": True
              }),
          ])
      )
  )
  def testAnySparse(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    self.assertEqual(sparse.any_sparse(
        test_case["classes"]), test_case["expected"])

  @combinations.generate(test_base.default_test_combinations())
  def assertShapesEqual(self, a, b):
    for a, b in zip(nest.flatten(a), nest.flatten(b)):
      self.assertEqual(a.ndims, b.ndims)
      if a.ndims is None:
        continue
      for c, d in zip(a.as_list(), b.as_list()):
        self.assertEqual(c, d)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject("Case_0", {
                  "types": (),
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", {
                  "types": tensor_shape.TensorShape([]),
                  "classes": ops.Tensor,
                  "expected": tensor_shape.TensorShape([])
              }),
              combinations.NamedObject("Case_2", {
                  "types": tensor_shape.TensorShape([]),
                  "classes": sparse_tensor.SparseTensor,
                  "expected": tensor_shape.unknown_shape()
              }),
              combinations.NamedObject("Case_3", {
                  "types": (tensor_shape.TensorShape([])),
                  "classes": (ops.Tensor),
                  "expected": (tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_4", {
                  "types": (tensor_shape.TensorShape([])),
                  "classes": (sparse_tensor.SparseTensor),
                  "expected": (tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_5", {
                  "types": (tensor_shape.TensorShape([]), ()),
                  "classes": (ops.Tensor, ()),
                  "expected": (tensor_shape.TensorShape([]), ())
              }),
              combinations.NamedObject("Case_6", {
                  "types": ((), tensor_shape.TensorShape([])),
                  "classes": ((), ops.Tensor),
                  "expected": ((), tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_7", {
                  "types": (tensor_shape.TensorShape([]), ()),
                  "classes": (sparse_tensor.SparseTensor, ()),
                  "expected": (tensor_shape.unknown_shape(), ())
              }),
              combinations.NamedObject("Case_8", {
                  "types": ((), tensor_shape.TensorShape([])),
                  "classes": ((), sparse_tensor.SparseTensor),
                  "expected": ((), tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_9", {
                  "types": (tensor_shape.TensorShape([]), (),
                            tensor_shape.TensorShape([])),
                  "classes": (ops.Tensor, (), ops.Tensor),
                  "expected": (tensor_shape.TensorShape([]), (),
                               tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_10", {
                  "types": (tensor_shape.TensorShape([]), (),
                            tensor_shape.TensorShape([])),
                  "classes": (sparse_tensor.SparseTensor, (),
                              sparse_tensor.SparseTensor),
                  "expected": (tensor_shape.unknown_shape(), (),
                               tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_11", {
                  "types": ((), tensor_shape.TensorShape([]), ()),
                  "classes": ((), ops.Tensor, ()),
                  "expected": ((), tensor_shape.TensorShape([]), ())
              }),
              combinations.NamedObject("Case_12", {
                  "types": ((), tensor_shape.TensorShape([]), ()),
                  "classes": ((), sparse_tensor.SparseTensor, ()),
                  "expected": ((), tensor_shape.unknown_shape(), ())
              }),
          ])
      )
  )
  def testAsDenseShapes(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    self.assertShapesEqual(
        sparse.as_dense_shapes(test_case["types"], test_case["classes"]),
        test_case["expected"])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject("Case_0", {
                  "types": (),
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", {
                  "types": dtypes.int32,
                  "classes": ops.Tensor,
                  "expected": dtypes.int32
              }),
              combinations.NamedObject("Case_2", {
                  "types": dtypes.int32,
                  "classes": sparse_tensor.SparseTensor,
                  "expected": dtypes.variant
              }),
              combinations.NamedObject("Case_3", {
                  "types": (dtypes.int32),
                  "classes": (ops.Tensor),
                  "expected": (dtypes.int32)
              }),
              combinations.NamedObject("Case_4", {
                  "types": (dtypes.int32),
                  "classes": (sparse_tensor.SparseTensor),
                  "expected": (dtypes.variant)
              }),
              combinations.NamedObject("Case_5", {
                  "types": (dtypes.int32, ()),
                  "classes": (ops.Tensor, ()),
                  "expected": (dtypes.int32, ())
              }),
              combinations.NamedObject("Case_6", {
                  "types": ((), dtypes.int32),
                  "classes": ((), ops.Tensor),
                  "expected": ((), dtypes.int32)
              }),
              combinations.NamedObject("Case_7", {
                  "types": (dtypes.int32, ()),
                  "classes": (sparse_tensor.SparseTensor, ()),
                  "expected": (dtypes.variant, ())
              }),
              combinations.NamedObject("Case_8", {
                  "types": ((), dtypes.int32),
                  "classes": ((), sparse_tensor.SparseTensor),
                  "expected": ((), dtypes.variant)
              }),
              combinations.NamedObject("Case_9", {
                  "types": (dtypes.int32, (), dtypes.int32),
                  "classes": (ops.Tensor, (), ops.Tensor),
                  "expected": (dtypes.int32, (), dtypes.int32)
              }),
              combinations.NamedObject("Case_10", {
                  "types": (dtypes.int32, (), dtypes.int32),
                  "classes": (sparse_tensor.SparseTensor, (),
                              sparse_tensor.SparseTensor),
                  "expected": (dtypes.variant, (), dtypes.variant)
              }),
              combinations.NamedObject("Case_11", {
                  "types": ((), dtypes.int32, ()),
                  "classes": ((), ops.Tensor, ()),
                  "expected": ((), dtypes.int32, ())
              }),
              combinations.NamedObject("Case_12", {
                  "types": ((), dtypes.int32, ()),
                  "classes": ((), sparse_tensor.SparseTensor, ()),
                  "expected": ((), dtypes.variant, ())
              }),
          ])
      )
  )
  def testAsDenseTypes(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    self.assertEqual(
        sparse.as_dense_types(test_case["types"], test_case["classes"]),
        test_case["expected"])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject("Case_0", {
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", {
                  "classes": sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]
                  ),
                  "expected": sparse_tensor.SparseTensor
              }),
              combinations.NamedObject("Case_2", {
                  "classes": constant_op.constant([1]),
                  "expected": ops.Tensor
              }),
              combinations.NamedObject("Case_3", {
                  "classes": (sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1])),
                  "expected": (sparse_tensor.SparseTensor)
              }),
              combinations.NamedObject("Case_4", {
                  "classes": (constant_op.constant([1])),
                  "expected": (ops.Tensor)
              }),
              combinations.NamedObject("Case_5", {
                  "classes": (sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]), ()),
                  "expected": (sparse_tensor.SparseTensor, ())
              }),
              combinations.NamedObject("Case_6", {
                  "classes": ((), sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1])),
                  "expected": ((), sparse_tensor.SparseTensor)
              }),
              combinations.NamedObject("Case_7", {
                  "classes": (constant_op.constant([1]), ()),
                  "expected": (ops.Tensor, ())
              }),
              combinations.NamedObject("Case_8", {
                  "classes": ((), constant_op.constant([1])),
                  "expected": ((), ops.Tensor)
              }),
              combinations.NamedObject("Case_9", {
                  "classes": (
                      sparse_tensor.SparseTensor(
                          indices=[[0]], values=[1], dense_shape=[1]),
                      (), constant_op.constant([1])),
                  "expected": (sparse_tensor.SparseTensor, (), ops.Tensor)
              }),
              combinations.NamedObject("Case_10", {
                  "classes": ((), sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]), ()),
                  "expected": ((), sparse_tensor.SparseTensor, ())
              }),
              combinations.NamedObject("Case_11", {
                  "classes": ((), constant_op.constant([1]), ()),
                  "expected": ((), ops.Tensor, ())
              }),
          ])
      )
  )
  def testGetClasses(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    self.assertEqual(
        sparse.get_classes(test_case["classes"]), test_case["expected"])

  @combinations.generate(test_base.default_test_combinations())
  def assertSparseValuesEqual(self, a, b):
    if not isinstance(a, sparse_tensor.SparseTensor):
      self.assertFalse(isinstance(b, sparse_tensor.SparseTensor))
      self.assertEqual(a, b)
      return
    self.assertTrue(isinstance(b, sparse_tensor.SparseTensor))
    with self.cached_session():
      self.assertAllEqual(a.eval().indices, self.evaluate(b).indices)
      self.assertAllEqual(a.eval().values, self.evaluate(b).values)
      self.assertAllEqual(a.eval().dense_shape, self.evaluate(b).dense_shape)

  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject(
                  "Case_0", ()),
              combinations.NamedObject(
                  "Case_1", sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1])),
              combinations.NamedObject(
                  "Case_2", sparse_tensor.SparseTensor(
                      indices=[[3, 4]], values=[-1], dense_shape=[4, 5])),
              combinations.NamedObject(
                  "Case_3", sparse_tensor.SparseTensor(
                      indices=[[0, 0], [3, 4]], values=[1, -1],
                      dense_shape=[4, 5])),
              combinations.NamedObject(
                  "Case_4", (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))),
              combinations.NamedObject(
                  "Case_5", (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]), ())),
              combinations.NamedObject(
                  "Case_6", ((), sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))),
          ])
      )
  )
  def testSerializeDeserialize(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    classes = sparse.get_classes(test_case)
    shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None),
                                classes)
    types = nest.map_structure(lambda _: dtypes.int32, classes)
    actual = sparse.deserialize_sparse_tensors(
        sparse.serialize_sparse_tensors(test_case), types, shapes,
        sparse.get_classes(test_case))
    nest.assert_same_structure(test_case, actual)
    for a, e in zip(nest.flatten(actual), nest.flatten(test_case)):
      self.assertSparseValuesEqual(a, e)

  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(test_case=[
              combinations.NamedObject(
                  "Case_0", ()),
              combinations.NamedObject(
                  "Case_1", sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1])),
              combinations.NamedObject(
                  "Case_2", sparse_tensor.SparseTensor(
                      indices=[[3, 4]], values=[-1], dense_shape=[4, 5])),
              combinations.NamedObject(
                  "Case_3", sparse_tensor.SparseTensor(
                      indices=[[0, 0], [3, 4]], values=[1, -1],
                      dense_shape=[4, 5])),
              combinations.NamedObject(
                  "Case_4", (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))),
              combinations.NamedObject(
                  "Case_5", (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]), ())),
              combinations.NamedObject(
                  "Case_6", ((), sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))),
          ])
      )
  )
  def testSerializeManyDeserialize(self, test_case):
    test_case = test_case._obj  # pylint: disable=protected-access
    classes = sparse.get_classes(test_case)
    shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None),
                                classes)
    types = nest.map_structure(lambda _: dtypes.int32, classes)
    actual = sparse.deserialize_sparse_tensors(
        sparse.serialize_many_sparse_tensors(test_case), types, shapes,
        sparse.get_classes(test_case))
    nest.assert_same_structure(test_case, actual)
    for a, e in zip(nest.flatten(actual), nest.flatten(test_case)):
      self.assertSparseValuesEqual(a, e)


if __name__ == "__main__":
  test.main()
