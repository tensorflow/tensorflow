"""Additional tests not covered by existing tensor handle tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

class TensorHandleTest(tf.test.TestCase):

  def testHandle(self):
    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    def testHandleForType(tf_dtype):
      for use_gpu in [True, False]:
        if not self._checkHaveGpu0():
          continue
        with self.test_session(use_gpu=use_gpu) as sess:
          n = 3
          input_value = tf.ones((n, n), dtype=tf_dtype)
          handle1 = tf.get_session_handle(input_value)
          handle2 = tf.get_session_handle(input_value)
          py_handle1, py_handle2 = sess.run([handle1, handle2])
          holder1, tensor1 = tf.get_session_tensor(py_handle1.handle, tf_dtype)
          holder2, tensor2 = tf.get_session_tensor(py_handle2.handle, tf_dtype)
          tensor3 = tf.add(tensor1, tensor2)

          feed_dict = {holder1: py_handle1.handle, holder2: py_handle2.handle}
          tensor3_numpy = sess.run(tensor3, feed_dict=feed_dict)

          np_dtype = tf_dtype.as_numpy_dtype()
          self.assertAllEqual(tensor3_numpy, 2*np.ones((n, n), dtype=np_dtype))

    testHandleForType(tf.float16)
    testHandleForType(tf.int32)
    testHandleForType(tf.float32)
    testHandleForType(tf.int64)
    testHandleForType(tf.float64)

  def testHandleAddGpu(self):
    # Simple addition test that catches when TensorFlow is built with wrong
    # compute capability.

    dt = tf.float32
    sess = tf.Session()

    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    with tf.device("gpu:0"):
      val_op = tf.ones((), dtype=dt)
      handle_op = tf.get_session_handle(val_op)

      py_handle = sess.run(handle_op)
      tf_handle = py_handle.handle
      holder1, tensor1 = tf.get_session_tensor(tf_handle, dt)
      holder2, tensor2 = tf.get_session_tensor(tf_handle, dt)
      add_op = tf.add(tensor1, tensor2)
      result_handle_op = tf.get_session_handle(add_op)
      for _ in range(10):
        tf_result_handle = sess.run(result_handle_op,
                                    feed_dict={holder1: tf_handle,
                                               holder2: tf_handle})
        np_result = tf_result_handle.eval()
        if np_result < 1.9:
          print(np_result)
      self.assertEqual(np_result, 2)

      
  def testPlaceholderIssue(self):
    # Test for https://github.com/tensorflow/tensorflow/issues/2587
    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    config = tf.ConfigProto()
    with self.test_session(config=config) as sess:
      dtype = tf.float32
      for device in ["cpu:0", "gpu:0"]:
        if not self._checkHaveGpu0():
          continue
        with tf.device(device):
          a_const = tf.constant(1, dtype)

          a_handle = sess.run(tf.get_session_handle(a_const))
          b_holder, b_tensor = tf.get_session_tensor(a_handle.handle, dtype)
          b_numpy = sess.run(b_tensor, feed_dict={b_holder: a_handle.handle})
          assert b_numpy == 1

  def testHandleDeletion(self):
    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    dtype = tf.float32

    config = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=config)

    # initial values live on CPU
    with tf.device("/cpu:0"):
      one = tf.constant(1, dtype=dtype)
      one_handle = sess.run(tf.get_session_handle(one))
      x_handle = sess.run(tf.get_session_handle(one))

    # addition lives on GPU
    with tf.device("/gpu:0"):
      add_holder1, add_tensor1 = tf.get_session_tensor(one_handle.handle, dtype)
      add_holder2, add_tensor2 = tf.get_session_tensor(one_handle.handle, dtype)
      add_op = tf.add(add_tensor1, add_tensor2)
      add_output = tf.get_session_handle(add_op)


    # add 1 to tensor 20 times to exceed _DEAD_HANDLES_THRESHOLD
    for _ in range(20):
      x_handle = sess.run(add_output, feed_dict={add_holder1: one_handle.handle,
                                                 add_holder2: x_handle.handle})

  def _checkHaveGpu0(self):
    device_names = [d.name for d in device_lib.list_local_devices()]
    return("/gpu:0" in device_names)
    
  def _assertHaveGpu0(self):
    """Check that GPU0 is available."""

    device_names = [d.name for d in device_lib.list_local_devices()]
    self.assertTrue("/gpu:0" in device_names)

if __name__ == "__main__":
  tf.test.main()
