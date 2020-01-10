import tensorflow.python.platform

import numpy
import tensorflow as tf


class GenerateIdentityTensorTest(tf.test.TestCase):

  def _testDiagOp(self, diag, dtype, expected_ans, use_gpu=False,
                  expected_err_re=None):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.diag(tf.convert_to_tensor(diag.astype(dtype)))
      out = tf_ans.eval()
    self.assertAllClose(out, expected_ans)
    self.assertShapeEqual(expected_ans, tf_ans)

  def testEmptyTensor(self):
    x = numpy.array([])
    expected_ans = numpy.empty([0, 0])
    self._testDiagOp(x, numpy.int32, expected_ans)

  def testRankOneIntTensor(self):
    x = numpy.array([1, 2, 3])
    expected_ans = numpy.array(
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]])
    self._testDiagOp(x, numpy.int32, expected_ans)
    self._testDiagOp(x, numpy.int64, expected_ans)

  def testRankOneFloatTensor(self):
    x = numpy.array([1.1, 2.2, 3.3])
    expected_ans = numpy.array(
        [[1.1, 0, 0],
         [0, 2.2, 0],
         [0, 0, 3.3]])
    self._testDiagOp(x, numpy.float32, expected_ans)
    self._testDiagOp(x, numpy.float64, expected_ans)

  def testRankTwoIntTensor(self):
    x = numpy.array([[1, 2, 3], [4, 5, 6]])
    expected_ans = numpy.array(
        [[[[1, 0, 0], [0, 0, 0]],
          [[0, 2, 0], [0, 0, 0]],
          [[0, 0, 3], [0, 0, 0]]],
         [[[0, 0, 0], [4, 0, 0]],
          [[0, 0, 0], [0, 5, 0]],
          [[0, 0, 0], [0, 0, 6]]]])
    self._testDiagOp(x, numpy.int32, expected_ans)
    self._testDiagOp(x, numpy.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    x = numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    expected_ans = numpy.array(
        [[[[1.1, 0, 0], [0, 0, 0]],
          [[0, 2.2, 0], [0, 0, 0]],
          [[0, 0, 3.3], [0, 0, 0]]],
         [[[0, 0, 0], [4.4, 0, 0]],
          [[0, 0, 0], [0, 5.5, 0]],
          [[0, 0, 0], [0, 0, 6.6]]]])
    self._testDiagOp(x, numpy.float32, expected_ans)
    self._testDiagOp(x, numpy.float64, expected_ans)

  def testRankThreeFloatTensor(self):
    x = numpy.array([[[1.1, 2.2], [3.3, 4.4]],
                     [[5.5, 6.6], [7.7, 8.8]]])
    expected_ans = numpy.array(
        [[[[[[1.1, 0], [0, 0]], [[0, 0], [0, 0]]],
           [[[0, 2.2], [0, 0]], [[0, 0], [0, 0]]]],
          [[[[0, 0], [3.3, 0]], [[0, 0], [0, 0]]],
           [[[0, 0], [0, 4.4]], [[0, 0], [0, 0]]]]],
         [[[[[0, 0], [0, 0]], [[5.5, 0], [0, 0]]],
           [[[0, 0], [0, 0]], [[0, 6.6], [0, 0]]]],
          [[[[0, 0], [0, 0]], [[0, 0], [7.7, 0]]],
           [[[0, 0], [0, 0]], [[0, 0], [0, 8.8]]]]]])
    self._testDiagOp(x, numpy.float32, expected_ans)
    self._testDiagOp(x, numpy.float64, expected_ans)

if __name__ == "__main__":
  tf.test.main()
