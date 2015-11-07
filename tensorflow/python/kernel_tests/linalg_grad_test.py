"""Tests for tensorflow.ops.linalg_grad."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class MatrixInverseGradientTest(tf.test.TestCase):
  pass  # Filled in below

def _GetMatrixInverseGradientTest(dtype, shape):
  def Test(self):
    with self.test_session():
      np.random.seed(1)
      m = np.random.uniform(low=1.0, high=100.0, size=np.prod(shape)).reshape(
          shape).astype(dtype)
      a = tf.constant(m)
      epsilon = np.finfo(dtype).eps
      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      delta = epsilon ** (1.0 / 3.0)
      tol = 1e-3

      if len(shape) == 2:
        ainv = tf.matrix_inverse(a)
      else:
        ainv = tf.batch_matrix_inverse(a)

      theoretical, numerical = gc.ComputeGradient(a, shape, ainv, shape,
                                                  delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)
  return Test


if __name__ == "__main__":
  # TODO(rmlarsen,irving): Reenable float32 once tolerances are fixed
  # The test used to loop over (np.float, np.double), both of which are float64.
  for dtype in np.float64,:
    for size in 2, 3, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(MatrixInverseGradientTest, 'testMatrixInverseGradient_' + name,
                _GetMatrixInverseGradientTest(dtype, shape))
  tf.test.main()
