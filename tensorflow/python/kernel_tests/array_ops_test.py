"""Tests for array_ops."""
import math

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class OperatorShapeTest(test_util.TensorFlowTestCase):

  def testExpandScalar(self):
    scalar = 'hello'
    scalar_expanded = array_ops.expand_dims(scalar, [0])
    self.assertEqual(scalar_expanded.get_shape(), (1,))

  def testSqueeze(self):
    scalar = 'hello'
    scalar_squeezed = array_ops.squeeze(scalar, ())
    self.assertEqual(scalar_squeezed.get_shape(), ())


class ReverseTest(test_util.TensorFlowTestCase):

  def testReverse0DimAuto(self):
    x_np = 4
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, []).eval()
        self.assertAllEqual(x_tf, x_np)

  def testReverse1DimAuto(self):
    x_np = [1, 4, 9]

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, [True]).eval()
        self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])


if __name__ == '__main__':
  googletest.main()
