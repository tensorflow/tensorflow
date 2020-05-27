# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for various tensorflow.ops.tf."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class ShapeOpsTest(test.TestCase):

  def _compareShape(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x)
      tf_ans_64 = array_ops.shape(x, out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeSparse(self, x_np, use_gpu=False):
    np_ans = np.array(np.shape(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeN(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu) as sess:
      tf_ans = array_ops.shape_n([x, x, x])
      tf_ans_64 = array_ops.shape_n([x, x, x], out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    for i in range(3):
      self.assertAllEqual(np_ans, result[i])
      self.assertAllEqual(np_ans, result_64[i])
      self.assertShapeEqual(np_ans, tf_ans[i])

  def _compareRank(self, x, use_gpu=False):
    np_ans = np.asarray(np.ndim(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareRankSparse(self, x_np, use_gpu=False):
    np_ans = np.asarray(np.ndim(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSize(self, x, use_gpu=False):
    np_ans = np.asarray(np.size(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x)
      result = self.evaluate(tf_ans)
      tf_ans_64 = array_ops.size(x, out_type=dtypes.int64)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSizeSparse(self, x_np, use_gpu=False):
    np_ans = np.asarray(np.size(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _testCpu(self, x):
    self._compareShape(x, use_gpu=False)
    self._compareShapeN(x, use_gpu=False)
    self._compareRank(x, use_gpu=False)
    self._compareSize(x, use_gpu=False)
    self._compareShapeSparse(x, use_gpu=False)
    self._compareRankSparse(x, use_gpu=False)
    self._compareSizeSparse(x, use_gpu=False)

  def _testGpu(self, x):
    self._compareShape(x, use_gpu=True)
    self._compareShapeN(x, use_gpu=True)
    self._compareRank(x, use_gpu=True)
    self._compareSize(x, use_gpu=True)
    self._compareShapeSparse(x, use_gpu=True)
    self._compareRankSparse(x, use_gpu=True)
    self._compareSizeSparse(x, use_gpu=True)

  def _testAll(self, x):
    self._testCpu(x)
    self._testGpu(x)

  def testBasic(self):
    self._testAll(np.random.randn(2))
    self._testAll(np.random.randn(2, 3))
    self._testAll(np.random.randn(2, 3, 5))
    self._testAll(np.random.randn(2, 3, 5, 7))
    self._testAll(np.random.randn(2, 3, 5, 7, 11))
    self._testAll(np.random.randn(2, 3, 5, 7, 11, 13))

  def testBool(self):
    self._testAll(np.random.choice((False, True), size=(2,)))
    self._testAll(np.random.choice((False, True), size=(2, 3)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7, 11)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7, 11, 13)))

  # Disabled because it takes too long to run, but manually verified
  # as passing at time of writing.
  def _test64BitOutput(self):
    with self.cached_session():
      inp = array_ops.zeros([2**31])
      num_elements = array_ops.size_internal(
          inp, optimize=False, out_type=dtypes.int64)
      self.assertEqual(2**31, self.evaluate(num_elements))

    # Too large for tf.int32 output.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      with self.cached_session():
        inp = array_ops.zeros([2**31])
        num_elements = array_ops.size_internal(
            inp, optimize=False, out_type=dtypes.int32)
        self.assertEqual(2**31, self.evaluate(num_elements))

  def _compareExpandDims(self, x, dim, use_gpu):
    np_ans = np.expand_dims(x, axis=dim)
    with self.cached_session(use_gpu=use_gpu):
      tensor = array_ops.expand_dims(x, dim)
      tf_ans = self.evaluate(tensor)
    self.assertShapeEqual(np_ans, tensor)
    self.assertAllEqual(np_ans, tf_ans)

  def _compareExpandDimsAll(self, x, dim):
    self._compareExpandDims(x, dim, False)
    self._compareExpandDims(x, dim, True)

  def testExpandDims(self):
    self._compareExpandDimsAll(np.zeros([2]), 0)
    self._compareExpandDimsAll(np.zeros([2]), 1)
    self._compareExpandDimsAll(np.zeros([2]), -1)

    self._compareExpandDimsAll(np.zeros([2, 3]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3]), -2)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 3)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -3)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -4)

  def testExpandDimsBool(self):
    choice = lambda s: np.random.choice((False, True), size=s)
    self._compareExpandDimsAll(choice([2]), 0)
    self._compareExpandDimsAll(choice([2]), 1)
    self._compareExpandDimsAll(choice([2]), -1)

    self._compareExpandDimsAll(choice([2, 3]), 0)
    self._compareExpandDimsAll(choice([2, 3]), 1)
    self._compareExpandDimsAll(choice([2, 3]), 2)
    self._compareExpandDimsAll(choice([2, 3]), -1)
    self._compareExpandDimsAll(choice([2, 3]), -2)

    self._compareExpandDimsAll(choice([2, 3, 5]), 0)
    self._compareExpandDimsAll(choice([2, 3, 5]), 1)
    self._compareExpandDimsAll(choice([2, 3, 5]), 2)
    self._compareExpandDimsAll(choice([2, 3, 5]), 3)

    self._compareExpandDimsAll(choice([2, 3, 5]), -1)
    self._compareExpandDimsAll(choice([2, 3, 5]), -2)
    self._compareExpandDimsAll(choice([2, 3, 5]), -3)
    self._compareExpandDimsAll(choice([2, 3, 5]), -4)

  @test_util.run_deprecated_v1
  def testExpandDimsErrors(self):
    with self.cached_session():
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), 4)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], 4)

  @test_util.run_deprecated_v1
  def testExpandDimsGradient(self):
    with self.cached_session():
      inp = constant_op.constant(
          np.random.rand(4, 2).astype("f"), dtype=dtypes.float32)
      squeezed = array_ops.expand_dims(inp, 1)

      err = gradient_checker.compute_gradient_error(inp, [4, 2], squeezed,
                                                    [4, 1, 2])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testExpandDimsScalar(self):
    with self.cached_session():
      inp = constant_op.constant(7)
      self.assertAllEqual([7], array_ops.expand_dims(inp, 0).eval())
      self.assertAllEqual([7], array_ops.expand_dims(inp, -1).eval())

      inp = constant_op.constant(True)
      self.assertAllEqual([True], array_ops.expand_dims(inp, 0).eval())
      self.assertAllEqual([True], array_ops.expand_dims(inp, -1).eval())

  def testExpandDimsDimType(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      x = np.zeros([2])
      np_ans = np.expand_dims(x, axis=0)
      with self.cached_session(use_gpu=True):
        tensor = array_ops.expand_dims(x, constant_op.constant(0, dtype))
        tf_ans = self.evaluate(tensor)
      self.assertShapeEqual(np_ans, tensor)
      self.assertAllEqual(np_ans, tf_ans)

  def _compareSqueeze(self, x, squeeze_dims, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      if squeeze_dims:
        np_ans = np.squeeze(x, axis=tuple(squeeze_dims))
        tensor = array_ops.squeeze(x, squeeze_dims)
        tf_ans = self.evaluate(tensor)
      else:
        np_ans = np.squeeze(x)
        tensor = array_ops.squeeze(x)
        tf_ans = self.evaluate(tensor)
    self.assertShapeEqual(np_ans, tensor)
    self.assertAllEqual(np_ans, tf_ans)

  def _compareSqueezeAll(self, x, squeeze_dims=None):
    if squeeze_dims is None:
      squeeze_dims = []
    self._compareSqueeze(x, squeeze_dims, False)
    self._compareSqueeze(x, squeeze_dims, True)

  def testSqueeze(self):
    # Nothing to squeeze.
    self._compareSqueezeAll(np.zeros([2]))
    self._compareSqueezeAll(np.zeros([2, 3]))

    # Squeeze the middle element away.
    self._compareSqueezeAll(np.zeros([2, 1, 2]))

    # Squeeze on both ends.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]))

  def testSqueezeBool(self):
    choice = lambda s: np.random.choice((False, True), size=s)
    # Nothing to squeeze.
    self._compareSqueezeAll(choice([2]))
    self._compareSqueezeAll(choice([2, 3]))

    # Squeeze the middle element away.
    self._compareSqueezeAll(choice([2, 1, 2]))

    # Squeeze on both ends.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]))

  def testSqueezeSpecificDimension(self):
    # Positive squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [2, 4])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0, 4, 2])

    # Negative squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-1])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5, -1])

  def testSqueezeSpecificDimensionBool(self):
    choice = lambda s: np.random.choice((False, True), size=s)
    # Positive squeeze dim index.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [0])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [2, 4])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [0, 4, 2])

    # Negative squeeze dim index.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-1])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-3, -5])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-3, -5, -1])

  def testSqueezeAllOnes(self):
    # Numpy squeezes a 1 element tensor into a zero dimensional tensor.
    # Verify that we do the same.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        tensor = array_ops.squeeze(np.zeros([1, 1, 1]), [])
        self.assertEqual(np.shape(1), tensor.get_shape())
        tf_ans = self.evaluate(tensor)
        self.assertEqual(np.shape(1), tf_ans.shape)

  def testSqueezeAllOnesBool(self):
    # Numpy squeezes a 1 element tensor into a zero dimensional tensor.
    # Verify that we do the same.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        tensor = array_ops.squeeze([[[False]]], [])
        self.assertEqual(np.shape(1), tensor.get_shape())
        tf_ans = self.evaluate(tensor)
        self.assertEqual(np.shape(1), tf_ans.shape)

  @test_util.run_deprecated_v1
  def testSqueezeOnlyOnes(self):
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        input_1x1x3 = np.zeros([1, 1, 3])
        self._compareSqueezeAll(input_1x1x3)
        self._compareSqueezeAll(input_1x1x3, [0])
        self._compareSqueezeAll(input_1x1x3, [1])
        self.assertRaises(ValueError, array_ops.squeeze, input_1x1x3, [2])

  @test_util.run_deprecated_v1
  def testSqueezeErrors(self):
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [-4])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [0, -4])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [3])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [2, 3])

  @test_util.run_deprecated_v1
  def testSqueezeGradient(self):
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = array_ops.reshape(inp, [4, 1, 2])
      squeezed = array_ops.squeeze(a, [])

      err = gradient_checker.compute_gradient_error(a, [4, 1, 2], squeezed,
                                                    [4, 2])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testSqueezeGradientWithSqueezeDims(self):
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = array_ops.reshape(inp, [4, 1, 2, 1])
      squeezed = array_ops.squeeze(a, [1])

      err = gradient_checker.compute_gradient_error(a, [4, 1, 2, 1], squeezed,
                                                    [4, 2, 1])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testSqueezeWithUnknownShape(self):
    with self.cached_session():
      a = array_ops.placeholder(dtypes.float32, shape=[2, None])

      squeezed = array_ops.squeeze(a, [1])
      self.assertEqual([2], squeezed.get_shape().as_list())

      squeezed = array_ops.squeeze(a)
      self.assertEqual(None, squeezed.get_shape())

      self.assertRaises(ValueError, array_ops.squeeze, a, [0])
      self.assertRaises(ValueError, array_ops.squeeze, a, [100])


class TileTest(test.TestCase, parameterized.TestCase):

  def testScalar(self):
    for use_gpu in False, True:
      with self.cached_session(use_gpu=use_gpu):
        a = constant_op.constant(7, shape=[], dtype=dtypes.float32)
        tiled = array_ops.tile(a, [])
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, ())
      self.assertEqual([], tiled.get_shape())
      self.assertEqual(7, result)

  def testSimple(self):
    # multiples could be int32 or int64
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(use_gpu=True):
        inp = np.random.rand(4, 1).astype(np.float32)
        a = constant_op.constant(inp)
        tiled = array_ops.tile(a, constant_op.constant([1, 4], dtype=dtype))
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, (4, 4))
      self.assertEqual([4, 4], tiled.get_shape())
      self.assertTrue((result == np.tile(inp, (1, 4))).all())

  def testIdentityTileAndGrad(self):
    with self.cached_session():
      inp = np.random.rand(4, 1).astype(np.float32)
      a = constant_op.constant(inp)
      tiled = array_ops.tile(a, [1, 1])
      result = self.evaluate(tiled)
    self.assertEqual(result.shape, (4, 1))
    self.assertEqual([4, 1], tiled.get_shape())
    self.assertTrue((result == np.tile(inp, (1, 1))).all())

  def testEmpty(self):
    with self.cached_session():
      inp = np.random.rand(2, 3).astype(np.float32)
      a = constant_op.constant(inp)
      tiled = array_ops.tile(a, [5, 0])
      result = self.evaluate(tiled)
    self.assertEqual(result.shape, (10, 0))
    self.assertEqual([10, 0], tiled.get_shape())

  @test_util.run_deprecated_v1
  def testUnknownInputShape(self):
    """Importing can call _TileShape without shape of <multiples> known."""
    with self.cached_session():
      inp = array_ops.placeholder(dtypes.float32)  # unknown shape
      multiples = constant_op.constant([1, 2, 3, 4], dtype=np.int32)
      tiled = array_ops.tile(inp, multiples)
      gdef = tiled.graph.as_graph_def()

      # Move the tile op to the start of the graph so that shapes of its inputs
      # are not available when the shape function runs on import.
      swapped = False
      for i, n in enumerate(gdef.node):
        if n.op == "Tile":
          # Swap tile op to be first in gdef.node
          assert i != 0
          new_node = node_def_pb2.NodeDef()
          new_node.CopyFrom(gdef.node[i])
          gdef.node[i].CopyFrom(gdef.node[0])
          gdef.node[0].CopyFrom(new_node)
          swapped = True
      assert swapped

      tiled_imported, = importer.import_graph_def(
          gdef, return_elements=[tiled.name])
      self.assertEqual(4, tiled_imported.get_shape().ndims)

  def testTypes(self):
    types_to_test = {
        "bool": (dtypes.bool, bool),
        "float32": (dtypes.float32, float),
        "float64": (dtypes.float64, float),
        "complex64": (dtypes.complex64, complex),
        "complex128": (dtypes.complex128, complex),
        "uint8": (dtypes.uint8, int),
        "int8": (dtypes.int8, int),
        "int16": (dtypes.int16, int),
        "int32": (dtypes.int32, int),
        "int64": (dtypes.int64, int),
        "uint32": (dtypes.uint32, int),
        "uint64": (dtypes.uint64, int),
        bytes: (dtypes.string, bytes)
    }
    for dtype_np, (dtype_tf, cast) in types_to_test.items():
      with self.cached_session(use_gpu=True):
        inp = np.random.rand(4, 1).astype(dtype_np)
        a = constant_op.constant(
            [cast(x) for x in inp.ravel(order="C")],
            shape=[4, 1],
            dtype=dtype_tf)
        tiled = array_ops.tile(a, [1, 4])
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, (4, 4))
      self.assertEqual([4, 4], tiled.get_shape())
      self.assertAllEqual(result, np.tile(inp, (1, 4)))

  @test_util.run_deprecated_v1
  def testInvalidDim(self):
    with self.cached_session():
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=[4, 1],
          dtype=dtypes.float32)
      # Wrong length of multiples.
      with self.assertRaises(ValueError):
        array_ops.tile(a, [1, 4, 2])
      # Wrong rank for multiples.
      with self.assertRaises(ValueError):
        array_ops.tile(a, [[2, 3], [3, 4]]).eval()

  def _RunAndVerifyResult(self, rank, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      # Random dims of given rank
      input_shape = np.random.randint(1, 4, size=rank)
      inp = np.random.rand(*input_shape).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=input_shape,
          dtype=dtypes.float32)
      multiples = np.random.randint(1, 4, size=rank).astype(np.int32)
      tiled = array_ops.tile(a, multiples)
      result = self.evaluate(tiled)
    self.assertTrue((np.array(multiples) * np.array(inp.shape) == np.array(
        result.shape)).all())
    self.assertAllEqual(result, np.tile(inp, tuple(multiples)))
    self.assertShapeEqual(result, tiled)

  def testRandom(self):
    # test low rank, like 5
    for _ in range(5):
      self._RunAndVerifyResult(5, use_gpu=False)
    for _ in range(5):
      self._RunAndVerifyResult(5, use_gpu=True)
    # test high rank, like 10
    for _ in range(5):
      self._RunAndVerifyResult(10, use_gpu=False)
    for _ in range(5):
      self._RunAndVerifyResult(10, use_gpu=True)

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  @test_util.run_deprecated_v1
  def testGradientSimpleReduction(self, multiples_dtype):
    with self.cached_session():
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 1], dtype=dtypes.float32)
      multiples = constant_op.constant([1, 4], dtype=multiples_dtype)
      tiled = array_ops.tile(a, multiples)
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = self.evaluate(grad)
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientStridedReduction(self):
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = self.evaluate(grad)
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertTrue((np.abs(expected - result) < 1e-3).all())

  @test_util.run_deprecated_v1
  def testGradientSimpleReductionOnGPU(self):
    with self.session(use_gpu=True):
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 1], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 4])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      result = self.evaluate(grad)
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientStridedReductionOnGPU(self):
    with self.session(use_gpu=True):
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      result = self.evaluate(grad)
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertAllClose(expected, result, 1e-3)

  def _RunAndVerifyGradientResult(self, input_shape, multiples):
    for use_gpu in False, True:
      with self.cached_session(use_gpu=use_gpu):
        # Random values
        inp = np.asarray(np.random.rand(*input_shape))
        a = constant_op.constant(inp, dtype=dtypes.float64)
        tiled = array_ops.tile(a, multiples)
        grad_shape = list(np.array(multiples) * np.array(inp.shape))
        err = gradient_checker.compute_gradient_error(
            a, list(input_shape), tiled, grad_shape, x_init_value=inp)
      print("tile(float) error = ", err)
      self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientRandomScalar(self):
    self._RunAndVerifyGradientResult([], [])

  @test_util.run_deprecated_v1
  def testGradientRandom(self):
    self._RunAndVerifyGradientResult([2, 2, 1, 1, 3], [1, 1, 1, 1, 1])
    self._RunAndVerifyGradientResult([2, 2, 1, 1, 3], [1, 2, 1, 3, 1])
    self._RunAndVerifyGradientResult([2, 3, 1, 1, 3], [3, 1, 1, 2, 2])
    self._RunAndVerifyGradientResult([2, 1, 3, 3, 2], [1, 3, 3, 1, 2])

  @test_util.run_deprecated_v1
  def testGradientStridedReductionGC(self):
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      err = gradient_checker.compute_gradient_error(a, [4, 2], tiled, [4, 4])
    self.assertLess(err, 1e-3)

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  @test_util.run_deprecated_v1
  def testGradientWithSparseGradWithRank1(self, multiples_dtype):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    multiples = constant_op.constant([3], dtype=dtypes.int64)
    outputs = array_ops.gather(array_ops.tile(inputs, multiples),
                               [1, 5, 9, 3, 7, 2, 2, 2])
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithSparseGradWithRank3(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    inputs = array_ops.reshape(inputs, [-1, 1, 1])
    outputs = array_ops.gather(array_ops.tile(inputs, [3, 4, 2]),
                               [1, 5, 9, 3, 7, 2, 2, 2])
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # Unknown multiples shape.
    inp = constant_op.constant(0.0, shape=[4, 4, 4, 4])
    tiled = array_ops.tile(inp, array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input shape.
    inp = array_ops.placeholder(dtypes.float32)
    tiled = array_ops.tile(inp, [2, 2, 2, 2])
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input and multiples shape.
    inp = array_ops.placeholder(dtypes.float32)
    tiled = array_ops.tile(inp, array_ops.placeholder(dtypes.int32))
    self.assertIs(None, tiled.get_shape().ndims)

    # Known input and partially known multiples.
    inp = constant_op.constant(0.0, shape=[1, 1])
    tiled = array_ops.tile(inp, [array_ops.placeholder(dtypes.int32), 7])
    self.assertEqual([None, 7], tiled.get_shape().as_list())

    # Mismatched input rank and multiples length.
    inp = array_ops.placeholder(dtypes.float32, shape=[None, None])
    with self.assertRaises(ValueError):
      tiled = array_ops.tile(
          inp, array_ops.placeholder(
              dtypes.int32, shape=[3]))


if __name__ == "__main__":
  test.main()
