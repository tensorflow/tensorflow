"""Tests for tensorflow.ops.reverse_sequence_op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class WhereOpTest(tf.test.TestCase):

  def _testWhere(self, x, truth, expected_err_re=None):
    with self.test_session():
      ans = tf.where(x)
      self.assertEqual([None, x.ndim], ans.get_shape().as_list())
      if expected_err_re is None:
        tf_ans = ans.eval()
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval()

  def testBasicMat(self):
    x = np.asarray([[True, False], [True, False]])

    # Ensure RowMajor mode
    truth = np.asarray([[0, 0], [1, 0]], dtype=np.int64)

    self._testWhere(x, truth)

  def testBasic3Tensor(self):
    x = np.asarray(
        [[[True, False], [True, False]], [[False, True], [False, True]],
         [[False, False], [False, True]]])

    # Ensure RowMajor mode
    truth = np.asarray(
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [2, 1, 1]],
        dtype=np.int64)

    self._testWhere(x, truth)


if __name__ == "__main__":
  tf.test.main()
