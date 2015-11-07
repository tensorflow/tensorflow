"""Tests for tensorflow.ops.math_ops."""
import math

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

exp = math.exp
log = math.log

class ReduceTest(test_util.TensorFlowTestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with self.test_session():
      y_tf = math_ops.reduce_sum(x).eval()
      self.assertEqual(y_tf, 21)

class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = [0.49, 0.7, -0.3, -0.8]
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu):
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.round(x_tf)
          y_tf_np = y_tf.eval()
          y_np = np.round(x_np)
          self.assertAllClose(y_tf_np, y_np, atol=1e-2)


class ModTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)

if __name__ == "__main__":
  googletest.main()
