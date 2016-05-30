# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

#from tensorflow.contrib.immediate.python.immediate import test_util

class EnvTest(tf.test.TestCase):

  def testHandle(self):
    def testHandleForType(tf_dtype):
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu) as sess:
          n = 3
          input_value = tf.ones((n, n), dtype=tf_dtype)
          handle1 = tf.get_session_handle(input_value)
          handle2 = tf.get_session_handle(input_value)
          holder1, tensor1 = tf.get_session_tensor(tf_dtype)
          holder2, tensor2 = tf.get_session_tensor(tf_dtype)
          tensor3 = tf.add(tensor1, tensor2)

          py_handle1, py_handle2 = sess.run([handle1, handle2])
          feed_dict = {holder1: py_handle1.handle, holder2: py_handle2.handle}
          tensor3_numpy = sess.run(tensor3, feed_dict=feed_dict)

          np_dtype = tf_dtype.as_numpy_dtype()
          self.assertAllEqual(tensor3_numpy, 2*np.ones((n, n), dtype=np_dtype))

    testHandleForType(tf.float16)
    testHandleForType(tf.int32)
    testHandleForType(tf.float32)
    testHandleForType(tf.int64)
    testHandleForType(tf.float64)

if __name__ == "__main__":
  tf.test.main()
