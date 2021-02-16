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

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


# NOTE(vikoth18): Arguments of parameterized tests are lifted into lambdas to make
# sure they are not executed before the (eager- or graph-mode) test environment
# has been set up.
#
class SparseTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case_fn=[
              combinations.NamedObject("Case_0", lambda: {
                  "classes": (),
                  "expected": False
              }),
              combinations.NamedObject("Case_1", lambda: {
                  "classes": (ops.Tensor),
                  "expected": False
              }),
              combinations.NamedObject("Case_2", lambda: {
                  "classes": (((ops.Tensor))),
                  "expected": False
              }),
              combinations.NamedObject("Case_3", lambda: {
                  "classes": (ops.Tensor, ops.Tensor),
                  "expected": False
              }),
              combinations.NamedObject("Case_4", lambda: {
                  "classes": (ops.Tensor, sparse_tensor.SparseTensor),
                  "expected": True
              }),
              combinations.NamedObject("Case_5", lambda: {
                  "classes": (sparse_tensor.SparseTensor,
                              sparse_tensor.SparseTensor),
                  "expected": True
              }),
              combinations.NamedObject("Case_6", lambda: {
                  "classes": (((sparse_tensor.SparseTensor))),
                  "expected": True
              }),
          ])
      )
  )
  def testAnySparse(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
    classes = test_case["classes"]
    expected = test_case["expected"]
    self.assertEqual(sparse.any_sparse(classes), expected)

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
          combinations.combine(test_case_fn=[
              combinations.NamedObject("Case_0", lambda: {
                  "types": (),
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", lambda: {
                  "types": tensor_shape.TensorShape([]),
                  "classes": ops.Tensor,
                  "expected": tensor_shape.TensorShape([])
              }),
              combinations.NamedObject("Case_2", lambda: {
                  "types": tensor_shape.TensorShape([]),
                  "classes": sparse_tensor.SparseTensor,
                  "expected": tensor_shape.unknown_shape()
              }),
              combinations.NamedObject("Case_3", lambda: {
                  "types": (tensor_shape.TensorShape([])),
                  "classes": (ops.Tensor),
                  "expected": (tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_4", lambda: {
                  "types": (tensor_shape.TensorShape([])),
                  "classes": (sparse_tensor.SparseTensor),
                  "expected": (tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_5", lambda: {
                  "types": (tensor_shape.TensorShape([]), ()),
                  "classes": (ops.Tensor, ()),
                  "expected": (tensor_shape.TensorShape([]), ())
              }),
              combinations.NamedObject("Case_6", lambda: {
                  "types": ((), tensor_shape.TensorShape([])),
                  "classes": ((), ops.Tensor),
                  "expected": ((), tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_7", lambda: {
                  "types": (tensor_shape.TensorShape([]), ()),
                  "classes": (sparse_tensor.SparseTensor, ()),
                  "expected": (tensor_shape.unknown_shape(), ())
              }),
              combinations.NamedObject("Case_8", lambda: {
                  "types": ((), tensor_shape.TensorShape([])),
                  "classes": ((), sparse_tensor.SparseTensor),
                  "expected": ((), tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_9", lambda: {
                  "types": (tensor_shape.TensorShape([]), (),
                            tensor_shape.TensorShape([])),
                  "classes": (ops.Tensor, (), ops.Tensor),
                  "expected": (tensor_shape.TensorShape([]), (),
                               tensor_shape.TensorShape([]))
              }),
              combinations.NamedObject("Case_10", lambda: {
                  "types": (tensor_shape.TensorShape([]), (),
                            tensor_shape.TensorShape([])),
                  "classes": (sparse_tensor.SparseTensor, (),
                              sparse_tensor.SparseTensor),
                  "expected": (tensor_shape.unknown_shape(), (),
                               tensor_shape.unknown_shape())
              }),
              combinations.NamedObject("Case_11", lambda: {
                  "types": ((), tensor_shape.TensorShape([]), ()),
                  "classes": ((), ops.Tensor, ()),
                  "expected": ((), tensor_shape.TensorShape([]), ())
              }),
              combinations.NamedObject("Case_12", lambda: {
                  "types": ((), tensor_shape.TensorShape([]), ()),
                  "classes": ((), sparse_tensor.SparseTensor, ()),
                  "expected": ((), tensor_shape.unknown_shape(), ())
              }),
          ])
      )
  )
  def testAsDenseShapes(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
    types = test_case["types"]
    classes = test_case["classes"]
    expected = test_case["expected"]
    self.assertShapesEqual(sparse.as_dense_shapes(types, classes), expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case_fn=[
              combinations.NamedObject("Case_0", lambda: {
                  "types": (),
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", lambda: {
                  "types": dtypes.int32,
                  "classes": ops.Tensor,
                  "expected": dtypes.int32
              }),
              combinations.NamedObject("Case_2", lambda: {
                  "types": dtypes.int32,
                  "classes": sparse_tensor.SparseTensor,
                  "expected": dtypes.variant
              }),
              combinations.NamedObject("Case_3", lambda: {
                  "types": (dtypes.int32),
                  "classes": (ops.Tensor),
                  "expected": (dtypes.int32)
              }),
              combinations.NamedObject("Case_4", lambda: {
                  "types": (dtypes.int32),
                  "classes": (sparse_tensor.SparseTensor),
                  "expected": (dtypes.variant)
              }),
              combinations.NamedObject("Case_5", lambda: {
                  "types": (dtypes.int32, ()),
                  "classes": (ops.Tensor, ()),
                  "expected": (dtypes.int32, ())
              }),
              combinations.NamedObject("Case_6", lambda: {
                  "types": ((), dtypes.int32),
                  "classes": ((), ops.Tensor),
                  "expected": ((), dtypes.int32)
              }),
              combinations.NamedObject("Case_7", lambda: {
                  "types": (dtypes.int32, ()),
                  "classes": (sparse_tensor.SparseTensor, ()),
                  "expected": (dtypes.variant, ())
              }),
              combinations.NamedObject("Case_8", lambda: {
                  "types": ((), dtypes.int32),
                  "classes": ((), sparse_tensor.SparseTensor),
                  "expected": ((), dtypes.variant)
              }),
              combinations.NamedObject("Case_9", lambda: {
                  "types": (dtypes.int32, (), dtypes.int32),
                  "classes": (ops.Tensor, (), ops.Tensor),
                  "expected": (dtypes.int32, (), dtypes.int32)
              }),
              combinations.NamedObject("Case_10", lambda: {
                  "types": (dtypes.int32, (), dtypes.int32),
                  "classes": (sparse_tensor.SparseTensor, (),
                              sparse_tensor.SparseTensor),
                  "expected": (dtypes.variant, (), dtypes.variant)
              }),
              combinations.NamedObject("Case_11", lambda: {
                  "types": ((), dtypes.int32, ()),
                  "classes": ((), ops.Tensor, ()),
                  "expected": ((), dtypes.int32, ())
              }),
              combinations.NamedObject("Case_12", lambda: {
                  "types": ((), dtypes.int32, ()),
                  "classes": ((), sparse_tensor.SparseTensor, ()),
                  "expected": ((), dtypes.variant, ())
              }),
          ])
      )
  )
  def testAsDenseTypes(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
    types = test_case["types"]
    classes = test_case["classes"]
    expected = test_case["expected"]
    self.assertEqual(sparse.as_dense_types(types, classes), expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(test_case_fn=[
              combinations.NamedObject("Case_0", lambda: {
                  "classes": (),
                  "expected": ()
              }),
              combinations.NamedObject("Case_1", lambda: {
                  "classes": sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]
                  ),
                  "expected": sparse_tensor.SparseTensor
              }),
              combinations.NamedObject("Case_2", lambda: {
                  "classes": constant_op.constant([1]),
                  "expected": ops.Tensor
              }),
              combinations.NamedObject("Case_3", lambda: {
                  "classes": (sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1])),
                  "expected": (sparse_tensor.SparseTensor)
              }),
              combinations.NamedObject("Case_4", lambda: {
                  "classes": (constant_op.constant([1])),
                  "expected": (ops.Tensor)
              }),
              combinations.NamedObject("Case_5", lambda: {
                  "classes": (sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]), ()),
                  "expected": (sparse_tensor.SparseTensor, ())
              }),
              combinations.NamedObject("Case_6", lambda: {
                  "classes": ((), sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1])),
                  "expected": ((), sparse_tensor.SparseTensor)
              }),
              combinations.NamedObject("Case_7", lambda: {
                  "classes": (constant_op.constant([1]), ()),
                  "expected": (ops.Tensor, ())
              }),
              combinations.NamedObject("Case_8", lambda: {
                  "classes": ((), constant_op.constant([1])),
                  "expected": ((), ops.Tensor)
              }),
              combinations.NamedObject("Case_9", lambda: {
                  "classes": (
                      sparse_tensor.SparseTensor(
                          indices=[[0]], values=[1], dense_shape=[1]),
                      (), constant_op.constant([1])),
                  "expected": (sparse_tensor.SparseTensor, (), ops.Tensor)
              }),
              combinations.NamedObject("Case_10", lambda: {
                  "classes": ((), sparse_tensor.SparseTensor(
                      indices=[[0]], values=[1], dense_shape=[1]), ()),
                  "expected": ((), sparse_tensor.SparseTensor, ())
              }),
              combinations.NamedObject("Case_11", lambda: {
                  "classes": ((), constant_op.constant([1]), ()),
                  "expected": ((), ops.Tensor, ())
              }),
          ])
      )
  )
  def testGetClasses(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
    classes = test_case["classes"]
    expected = test_case["expected"]
    self.assertEqual(sparse.get_classes(classes), expected)

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
          combinations.combine(test_case_fn=[
              combinations.NamedObject(
                  "Case_0", lambda: ()
              ),
              combinations.NamedObject(
                  "Case_1", lambda: sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1])
              ),
              combinations.NamedObject(
                  "Case_2", lambda: sparse_tensor.SparseTensor(
                      indices=[[3, 4]], values=[-1], dense_shape=[4, 5])
              ),
              combinations.NamedObject(
                  "Case_3", lambda: sparse_tensor.SparseTensor(
                      indices=[[0, 0], [3, 4]], values=[1, -1],
                      dense_shape=[4, 5])
              ),
              combinations.NamedObject(
                  "Case_4", lambda: (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))
              ),
              combinations.NamedObject(
                  "Case_5", lambda: (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]), ())
              ),
              combinations.NamedObject(
                  "Case_6", lambda: ((), sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))),
          ])
      )
  )
  def testSerializeDeserialize(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
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
          combinations.combine(test_case_fn=[
              combinations.NamedObject(
                  "Case_0", lambda: ()
              ),
              combinations.NamedObject(
                  "Case_1", lambda: sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1])
              ),
              combinations.NamedObject(
                  "Case_2", lambda: sparse_tensor.SparseTensor(
                      indices=[[3, 4]], values=[-1], dense_shape=[4, 5])
              ),
              combinations.NamedObject(
                  "Case_3", lambda: sparse_tensor.SparseTensor(
                      indices=[[0, 0], [3, 4]], values=[1, -1],
                      dense_shape=[4, 5])
              ),
              combinations.NamedObject(
                  "Case_4", lambda: (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))
              ),
              combinations.NamedObject(
                  "Case_5", lambda: (sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]), ())
              ),
              combinations.NamedObject(
                  "Case_6", lambda: ((), sparse_tensor.SparseTensor(
                      indices=[[0, 0]], values=[1], dense_shape=[1, 1]))
              ),
          ])
      )
  )
  def testSerializeManyDeserialize(self, test_case_fn):
    test_case_fn = test_case_fn._obj  # pylint: disable=protected-access
    test_case = test_case_fn()
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
