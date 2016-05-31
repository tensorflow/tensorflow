# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate
from tensorflow.contrib.immediate.python.immediate import test_util

class EnvTest(tf.test.TestCase):

  def testPlaceholderOnGpuIssue(self):
    # https://github.com/tensorflow/tensorflow/issues/2587
    config = tf.ConfigProto(log_device_placement=True)
    with self.test_session(config=config) as sess:
      dtype=tf.float32
      with tf.device("/gpu:0"):
        a = tf.constant(1, dtype)

      a_handle = sess.run(tf.get_session_handle(a))
      b_holder, b_tensor = tf.get_session_tensor(dtype)
      print(sess.run(b_tensor, feed_dict={b_holder:
                                          a_handle.handle}))

  def testSessionTensorOnGpuIssue(self):
    # https://github.com/tensorflow/tensorflow/issues/2586
    with self.test_session() as sess:                                           
      a = tf.constant(1.0)                                                      
      a_handle_op = tf.get_session_handle(a)                                    
      b = tf.constant(2.0)                                                      
      b_handle_op = tf.get_session_handle(b)                                    

      failure_case = True                                                       
      if failure_case:                                                          
        a_p, a_t = tf.get_session_tensor(tf.float32)                            
        b_p, b_t = tf.get_session_tensor(tf.float32)                            
        a_handle = sess.run(a_handle_op)                                        
        b_handle = sess.run(b_handle_op)                                        
      else:                                                                     
        a_handle = sess.run(a_handle_op)                                        
        b_handle = sess.run(b_handle_op)                                        
        a_p, a_t = tf.get_session_tensor(tf.float32)                            
        b_p, b_t = tf.get_session_tensor(tf.float32)                            

      c = tf.add(a_t, b_t)                                                      
      c_handle = sess.run(                                                      
        tf.get_session_handle(c),                                               
        feed_dict={a_p: a_handle.handle,                                        
                   b_p: b_handle.handle})                                       
      self.assertEqual(3.0, c_handle.eval())                                    

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
