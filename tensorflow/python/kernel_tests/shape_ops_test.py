# Copyright 2015 Google Inc. All Rights Reserved.
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

import numpy as np

import tensorflow as tf


class ShapeOpsTest(tf.test.TestCase):

  def _compareShape(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.shape(x)
      result = tf_ans.eval()
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeN(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.test_session(use_gpu=use_gpu) as sess:
      tf_ans = tf.shape_n([x, x, x])
      result = sess.run(tf_ans)
    for i in range(3):
      self.assertAllEqual(np_ans, result[i])
      self.assertShapeEqual(np_ans, tf_ans[i])

  def _compareRank(self, x, use_gpu=False):
    np_ans = np.asarray(np.ndim(x))
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.rank(x)
      result = tf_ans.eval()
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSize(self, x, use_gpu=False):
    np_ans = np.asarray(np.size(x))
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.size(x)
      result = tf_ans.eval()
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _testCpu(self, x):
    self._compareShape(x, use_gpu=False)
    self._compareShapeN(x, use_gpu=False)
    self._compareRank(x, use_gpu=False)
    self._compareSize(x, use_gpu=False)

  def _testGpu(self, x):
    self._compareShape(x, use_gpu=True)
    self._compareShapeN(x, use_gpu=True)
    self._compareRank(x, use_gpu=True)
    self._compareSize(x, use_gpu=True)

  def _testAll(self, x):
    self._testCpu(x)
    self._testGpu(x)

  def testBasic(self):
    self._testAll(np.zeros([2]))
    self._testAll(np.zeros([2, 3]))
    self._testAll(np.zeros([2, 3, 5]))
    self._testAll(np.zeros([2, 3, 5, 7]))
    self._testAll(np.zeros([2, 3, 5, 7, 11]))
    self._testAll(np.zeros([2, 3, 5, 7, 11, 13]))

  def _compareExpandDims(self, x, dim, use_gpu):
    np_ans = np.expand_dims(x, axis=dim)
    with self.test_session(use_gpu=use_gpu):
      tensor = tf.expand_dims(x, dim)
      tf_ans = tensor.eval()
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

  def testExpandDimsErrors(self):
    with self.test_session():
      self.assertRaises(ValueError, tf.expand_dims, np.zeros([2, 3, 5]), -5)
      self.assertRaises(ValueError, tf.expand_dims, np.zeros([2, 3, 5]), 4)

  def testExpandDimsGradient(self):
    with self.test_session():
      inp = tf.constant(np.random.rand(4, 2).astype("f"),
                     dtype=tf.float32)
      squeezed = tf.expand_dims(inp, 1)

      err = tf.test.compute_gradient_error(inp, [4, 2], squeezed, [4, 1, 2])
    self.assertLess(err, 1e-3)

  def testExpandDimsScalar(self):
    with self.test_session():
      inp = tf.constant(7)
      self.assertAllEqual([7], tf.expand_dims(inp, 0).eval())
      self.assertAllEqual([7], tf.expand_dims(inp, -1).eval())

  def _compareSqueeze(self, x, squeeze_dims, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      if squeeze_dims:
        np_ans = np.squeeze(x, axis=tuple(squeeze_dims))
        tensor = tf.squeeze(x, squeeze_dims)
        tf_ans = tensor.eval()
      else:
        np_ans = np.squeeze(x)
        tensor = tf.squeeze(x)
        tf_ans = tensor.eval()
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

  def testSqueezeSpecificDimension(self):
    # Positive squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [2, 4])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0, 4, 2])

    # Negative squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-1])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5, -1])

  def testSqueezeAllOnes(self):
    # Numpy squeezes a 1 element tensor into a zero dimensional tensor.
    # Verify that we do the same.
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        tensor = tf.squeeze(np.zeros([1, 1, 1]), [])
        self.assertEqual(np.shape(1), tensor.get_shape())
        tf_ans = tensor.eval()
        self.assertEqual(np.shape(1), tf_ans.shape)

  def testSqueezeOnlyOnes(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        input_1x1x3 = np.zeros([1, 1, 3])
        self._compareSqueezeAll(input_1x1x3)
        self._compareSqueezeAll(input_1x1x3, [0])
        self._compareSqueezeAll(input_1x1x3, [1])
        self.assertRaises(ValueError, tf.squeeze, input_1x1x3, [2])

  def testSqueezeErrors(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        self.assertRaises(ValueError, tf.squeeze, np.zeros([1, 2, 1]), [-4])
        self.assertRaises(ValueError, tf.squeeze, np.zeros([1, 2, 1]), [0, -4])
        self.assertRaises(ValueError, tf.squeeze, np.zeros([1, 2, 1]), [3])
        self.assertRaises(ValueError, tf.squeeze, np.zeros([1, 2, 1]), [2, 3])

  def testSqueezeGradient(self):
    with self.test_session():
      inp = np.random.rand(4, 2).astype("f")
      a = tf.reshape(inp, [4, 1, 2])
      squeezed = tf.squeeze(a, [])

      err = tf.test.compute_gradient_error(a, [4, 1, 2], squeezed, [4, 2])
    self.assertLess(err, 1e-3)

  def testSqueezeGradientWithSqueezeDims(self):
    with self.test_session():
      inp = np.random.rand(4, 2).astype("f")
      a = tf.reshape(inp, [4, 1, 2, 1])
      squeezed = tf.squeeze(a, [1])

      err = tf.test.compute_gradient_error(a, [4, 1, 2, 1], squeezed, [4, 2, 1])
    self.assertLess(err, 1e-3)

  def testSqueezeWithUnknownShape(self):
    with self.test_session():
      a = tf.placeholder(tf.float32, shape=[2, None])

      squeezed = tf.squeeze(a, [1])
      self.assertEqual([2], squeezed.get_shape().as_list())

      squeezed = tf.squeeze(a)
      self.assertEqual(None, squeezed.get_shape())

      self.assertRaises(ValueError, tf.squeeze, a, [0])
      self.assertRaises(ValueError, tf.squeeze, a, [100])


class TileTest(tf.test.TestCase):

  def testScalar(self):
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        a = tf.constant(7, shape=[], dtype=tf.float32)
        tiled = tf.tile(a, [])
        result = tiled.eval()
      self.assertEqual(result.shape, ())
      self.assertEqual([], tiled.get_shape())
      self.assertEqual(7, result)

  def testSimple(self):
    with self.test_session():
      inp = np.random.rand(4, 1).astype(np.float32)
      a = tf.constant(inp)
      tiled = tf.tile(a, [1, 4])
      result = tiled.eval()
    self.assertEqual(result.shape, (4, 4))
    self.assertEqual([4, 4], tiled.get_shape())
    self.assertTrue((result == np.tile(inp, (1, 4))).all())

  def testEmpty(self):
    with self.test_session():
      inp = np.random.rand(2, 3).astype(np.float32)
      a = tf.constant(inp)
      tiled = tf.tile(a, [5, 0])
      result = tiled.eval()
    self.assertEqual(result.shape, (10, 0))
    self.assertEqual([10, 0], tiled.get_shape())

  def testTypes(self):
    types_to_test = {
        "bool": (tf.bool, bool),
        "float32": (tf.float32, float),
        "float64": (tf.float64, float),
        "uint8": (tf.uint8, int),
        "int32": (tf.int32, int),
        "int64": (tf.int64, int),
        bytes: (tf.string, bytes)
    }
    for dtype_np, (dtype_tf, cast) in types_to_test.items():
      with self.test_session():
        inp = np.random.rand(4, 1).astype(dtype_np)
        a = tf.constant([cast(x) for x in inp.ravel(order="C")],
                     shape=[4, 1],
                     dtype=dtype_tf)
        tiled = tf.tile(a, [1, 4])
        result = tiled.eval()
      self.assertEqual(result.shape, (4, 4))
      self.assertEqual([4, 4], tiled.get_shape())
      self.assertAllEqual(result, np.tile(inp, (1, 4)))

  def testInvalidDim(self):
    with self.test_session():
      inp = np.random.rand(4, 1).astype("f")
      a = tf.constant([float(x) for x in inp.ravel(order="C")],
                   shape=[4, 1], dtype=tf.float32)
      # Wrong length of multiples.
      with self.assertRaises(ValueError):
        tf.tile(a, [1, 4, 2])
      # Wrong rank for multiples.
      with self.assertRaises(ValueError):
        tf.tile(a, [[2, 3], [3, 4]]).eval()

  def _RunAndVerifyResult(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      # Random dims of rank 5
      input_shape = np.random.randint(1, 4, size=5)
      inp = np.random.rand(*input_shape).astype("f")
      a = tf.constant([float(x) for x in inp.ravel(order="C")],
                   shape=input_shape, dtype=tf.float32)
      multiples = np.random.randint(1, 4, size=5).astype(np.int32)
      tiled = tf.tile(a, multiples)
      result = tiled.eval()
    self.assertTrue((np.array(multiples) * np.array(inp.shape) ==
                     np.array(result.shape)).all())
    self.assertAllEqual(result, np.tile(inp, tuple(multiples)))
    self.assertShapeEqual(result, tiled)

  def testRandom(self):
    for _ in range(5):
      self._RunAndVerifyResult(use_gpu=False)
    for _ in range(5):
      self._RunAndVerifyResult(use_gpu=True)

  def testGradientSimpleReduction(self):
    with self.test_session():
      inp = np.random.rand(4, 1).astype("f")
      a = tf.constant([float(x) for x in inp.flatten()],
                   shape=[4, 1], dtype=tf.float32)
      tiled = tf.tile(a, [1, 4])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                             shape=grad_shape)
      grad = tf.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = grad.eval()
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  def testGradientStridedReduction(self):
    with self.test_session():
      inp = np.random.rand(4, 2).astype("f")
      a = tf.constant([float(x) for x in inp.flatten()],
                   shape=[4, 2], dtype=tf.float32)
      tiled = tf.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                             shape=grad_shape)
      grad = tf.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = grad.eval()
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertTrue((np.abs(expected - result) < 1e-3).all())

  def testGradientSimpleReductionOnGPU(self):
    with self.test_session(use_gpu=True):
      inp = np.random.rand(4, 1).astype("f")
      a = tf.constant([float(x) for x in inp.flatten()],
                   shape=[4, 1], dtype=tf.float32)
      tiled = tf.tile(a, [1, 4])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                             shape=grad_shape)
      grad = tf.gradients([tiled], [a], [grad_tensor])[0]
      result = grad.eval()
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  def testGradientStridedReductionOnGPU(self):
    with self.test_session(use_gpu=True):
      inp = np.random.rand(4, 2).astype("f")
      a = tf.constant([float(x) for x in inp.flatten()],
                   shape=[4, 2], dtype=tf.float32)
      tiled = tf.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                             shape=grad_shape)
      grad = tf.gradients([tiled], [a], [grad_tensor])[0]
      result = grad.eval()
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertAllClose(expected, result, 1e-3)

  def _RunAndVerifyGradientResult(self, input_shape, multiples):
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        # Random values
        inp = np.asarray(np.random.rand(*input_shape))
        a = tf.constant(inp, dtype=tf.float64)
        tiled = tf.tile(a, multiples)
        grad_shape = list(np.array(multiples) * np.array(inp.shape))
        err = tf.test.compute_gradient_error(a,
                                             list(input_shape),
                                             tiled,
                                             grad_shape,
                                             x_init_value=inp)
      print("tile(float) error = ", err)
      self.assertLess(err, 1e-3)

  def testGradientRandomScalar(self):
    self._RunAndVerifyGradientResult([], [])

  def testGradientRandom(self):
    self._RunAndVerifyGradientResult([2, 2, 1, 1, 3], [1, 2, 1, 3, 1])
    self._RunAndVerifyGradientResult([2, 3, 1, 1, 3], [3, 1, 1, 2, 2])
    self._RunAndVerifyGradientResult([2, 1, 3, 3, 2], [1, 3, 3, 1, 2])

  def testGradientStridedReductionGC(self):
    with self.test_session():
      inp = np.random.rand(4, 2).astype("f")
      a = tf.constant([float(x) for x in inp.flatten()],
                   shape=[4, 2], dtype=tf.float32)
      tiled = tf.tile(a, [1, 2])
      err = tf.test.compute_gradient_error(a, [4, 2], tiled, [4, 4])
    self.assertLess(err, 1e-3)

  def testShapeFunctionEdgeCases(self):
    # Unknown multiples shape.
    inp = tf.constant(0.0, shape=[4, 4, 4, 4])
    tiled = tf.tile(inp, tf.placeholder(tf.int32))
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input shape.
    inp = tf.placeholder(tf.float32)
    tiled = tf.tile(inp, [2, 2, 2, 2])
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input and multiples shape.
    inp = tf.placeholder(tf.float32)
    tiled = tf.tile(inp, tf.placeholder(tf.int32))
    self.assertIs(None, tiled.get_shape().ndims)


if __name__ == "__main__":
  tf.test.main()
