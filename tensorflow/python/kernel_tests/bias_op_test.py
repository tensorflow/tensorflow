"""Functional tests for BiasAdd."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker


class BiasAddTest(tf.test.TestCase):

  def _npBias(self, inputs, bias):
    assert len(bias.shape) == 1
    print inputs.shape
    print bias.shape
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(([1] * (len(inputs.shape) - 1))
                                 + [bias.shape[0]])

  def testNpBias(self):
    self.assertAllClose(np.array([[11, 22, 33], [41, 52, 63]]),
                        self._npBias(np.array([[10, 20, 30], [40, 50, 60]]),
                                     np.array([1, 2, 3])))

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    np_val = self._npBias(np_inputs, np_bias)
    with self.test_session(use_gpu=use_gpu):
      tf_val = tf.nn.bias_add(np_inputs, np_bias).eval()
    self.assertAllClose(np_val, tf_val)

  def _testAll(self, np_inputs, np_bias):
    self._testBias(np_inputs, np_bias, use_gpu=False)
    if np_inputs.dtype == np.float32 or np_inputs.dtype == np.float64:
      self._testBias(np_inputs, np_bias, use_gpu=True)

  def testInputDims(self):
    with self.assertRaises(ValueError):
      tf.nn.bias_add([1, 2], [1])

  def testBiasVec(self):
    with self.assertRaises(ValueError):
      tf.nn.bias_add(tf.reshape([1, 2], shape=[1, 2]),
                      tf.reshape([1, 2], shape=[1, 2]))

  def testBiasInputsMatch(self):
    with self.assertRaises(ValueError):
      tf.nn.bias_add(tf.reshape([1, 2], shape=[1, 2]),
                      tf.reshape([1], shape=[1]))

  def testIntTypes(self):
    for t in [np.int8, np.int16, np.int32, np.int64]:
      self._testAll(np.array([[10, 20, 30], [40, 50, 60]]).astype(t),
                    np.array([1, 2, 3]).astype(t))

  def testFloatTypes(self):
    for t in [np.float32, np.float64]:
      self._testAll(np.random.rand(4, 3, 3).astype(t),
                    np.random.rand(3).astype(t))

  def testGradientTensor(self):
    with self.test_session():
      t = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                               dtype=tf.float64)
      b = tf.constant([1.3, 2.4], dtype=tf.float64)
      bo = tf.nn.bias_add(t, b)
      err = gradient_checker.ComputeGradientError(t, [3, 2], bo, [3, 2])
    print "bias add tensor gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradientBias(self):
    with self.test_session():
      t = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                               dtype=tf.float64)
      b = tf.constant([1.3, 2.4], dtype=tf.float64)
      bo = tf.nn.bias_add(t, b)
      err = gradient_checker.ComputeGradientError(b, [2], bo, [3, 2])
    print "bias add bias gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradientTensor4D(self):
    with self.test_session():
      s = [2, 3, 4, 2]
      x = np.arange(1.0, 49.0).reshape(s).astype(np.float32)
      t = tf.constant(x, shape=s, dtype=tf.float32)
      b = tf.constant([1.3, 2.4], dtype=tf.float32)
      bo = tf.nn.bias_add(t, b)
      err = gradient_checker.ComputeGradientError(t, s, bo, s, x_init_value=x)
    print "bias add tensor gradient err = ", err
    self.assertLess(err, 1e-3)


if __name__ == "__main__":
  tf.test.main()
