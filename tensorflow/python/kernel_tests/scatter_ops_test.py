"""Tests for tensorflow.ops.tf.scatter."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class ScatterTest(tf.test.TestCase):

  def _VariableRankTest(self, np_scatter, tf_scatter):
    np.random.seed(8)
    with self.test_session():
      for indices_shape in (), (2,), (2, 3), (2, 3, 4):
        for extra_shape in (), (5,), (5, 6):
          # Generate random indices with no duplicates for easy numpy comparison
          size = np.prod(indices_shape, dtype=np.int32)
          indices = np.arange(2 * size)
          np.random.shuffle(indices)
          indices = indices[:size].reshape(indices_shape)
          updates = np.random.randn(*(indices_shape + extra_shape))
          old = np.random.randn(*((2 * size,) + extra_shape))
        # Scatter via numpy
        new = old.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref = tf.Variable(old)
        ref.initializer.run()
        tf_scatter(ref, indices, updates).eval()
        # Compare
        self.assertAllClose(ref.eval(), new)

  def testVariableRankUpdate(self):
    def update(ref, indices, updates):
      ref[indices] = updates
    self._VariableRankTest(update, tf.scatter_update)

  def testVariableRankAdd(self):
    def add(ref, indices, updates):
      ref[indices] += updates
    self._VariableRankTest(add, tf.scatter_add)

  def testVariableRankSub(self):
    def sub(ref, indices, updates):
      ref[indices] -= updates
    self._VariableRankTest(sub, tf.scatter_sub)


if __name__ == "__main__":
  tf.test.main()
