from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.naturali.python.ops import lookahead_ops
from tensorflow.python.ops import variable_scope as vs

class LookaheadTest(tf.test.TestCase):
  _use_gpu = False
  def testLookaheadgrad(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      x1 = [[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],[[5.0,6.0,7.0,8.0],[5.0,6.0,7.0,8.0]],[[9.0,10.0,11.0,12.0],[9.0,10.0,11.0,12.0]]]
      x2 = [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]

      result = lookahead_ops.lookahead(x1,x2)
      self.assertAllEqual(result.eval(), [[[26.0,40.0,58.0,80.0],[26.0,40.0,58.0,80.0]],[[50.0,72.0,98.0,128.0],[50.0,72.0,98.0,128.0]],[[9.0,20.0,33.0,48.0],[9.0,20.0,33.0,48.0]]])
      self.assertAllEqual(result.eval(), [[[26.0,40.0,58.0,80.0],[26.0,40.0,58.0,80.0]],[[50.0,72.0,98.0,128.0],[50.0,72.0,98.0,128.0]],[[9.0,20.0,33.0,48.0],[9.0,20.0,33.0,48.0]]])
      self.assertAllEqual(tf.test.is_built_with_cuda(), 1)


  def testLookaheadgrad(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      x1 = tf.constant([[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],[[5.0,6.0,7.0,8.0],[5.0,6.0,7.0,8.0]],[[9.0,10.0,11.0,12.0],[9.0,10.0,11.0,12.0]]])
      x2 = tf.constant([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]])
      x3 = tf.constant([[[1.0,2.0,4.0,8.0],[1.0,2.0,4.0,8.0]],[[4.0,8.0,16.0,32.0],[4.0,8.0,16.0,32.0]],[[8.0,16.0,32.0,64.0],[8.0,16.0,32.0,64.0]]])
      result = lookahead_ops.lookaheadgrad(x1,x2,x3)

      r0 = [[[1.0,4.0,12.0,32.0],[1.0,4.0,12.0,32.0]],[[9.0,28.0,76.0,192.0],[9.0,28.0,76.0,192.0]],[[28.0,80.0,208.0,512.0],[28.0,80.0,208.0,512.0]]]
      r1 = [[186.0,424.0,952.0,2112.0],[82.0,184.0,408.0,896.0]]
      self.assertAllEqual(result[0].eval(), r0)
      self.assertAllEqual(result[1].eval(), r1) 

      xw = lookahead_ops.lookahead(x1, x2)
      w_grad = tf.gradients(xw, [x1, x2])
      self.assertAllEqual(w_grad[0].eval(), [[[1.,2.,3.,4.],[1.,2.,3.,4.]], [[  6.,   8.,  10.,  12.],[  6.,   8.,  10.,  12.]], [[  6.,   8.,  10.,  12.],[  6.,   8.,  10.,  12.]]])
      self.assertAllEqual(w_grad[1].eval(), [[30., 36., 42., 48.],[28., 32., 36., 40.]])

if __name__ == "__main__":
  tf.test.main()
