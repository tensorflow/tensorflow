
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LookaheadTest(tf.test.TestCase):

  def test(self):
    with self.test_session(): 
      x1 = [[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],[[5.0,6.0,7.0,8.0],[5.0,6.0,7.0,8.0]],[[9.0,10.0,11.0,12.0],[9.0,10.0,11.0,12.0]]]
      x2 = [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]
      result = tf.user_ops.lookahead(x1,x2)
      self.assertAllEqual(result.eval(), [[[26.0,40.0,58.0,80.0],[26.0,40.0,58.0,80.0]],[[50.0,72.0,98.0,128.0],[50.0,72.0,98.0,128.0]],[[9.0,20.0,33.0,48.0],[9.0,20.0,33.0,48.0]]])
      self.assertAllEqual(result.eval(), [[[26.0,40.0,58.0,80.0],[26.0,40.0,58.0,80.0]],[[50.0,72.0,98.0,128.0],[50.0,72.0,98.0,128.0]],[[9.0,20.0,33.0,48.0],[9.0,20.0,33.0,48.0]]])
      self.assertAllEqual(tf.test.is_built_with_cuda(), 1)
      if tf.test.is_built_with_cuda():
        with tf.device("/gpu:0"):
          result_2 = tf.user_ops.lookaheadgpu(x1,x2)
          self.assertAllEqual(result_2.eval(), [[[26.0,40.0,58.0,80.0],[26.0,40.0,58.0,80.0]],[[50.0,72.0,98.0,128.0],[50.0,72.0,98.0,128.0]],[[9.0,20.0,33.0,48.0],[9.0,20.0,33.0,48.0]]])

if __name__ == '__main__':
  tf.test.main()
