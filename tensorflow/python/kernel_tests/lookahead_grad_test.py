from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class LookaheadGradTest(tf.test.TestCase):

  def test(self):
    with self.test_session():
      x1 = [[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],[[5.0,6.0,7.0,8.0],[5.0,6.0,7.0,8.0]],[[9.0,10.0,11.0,12.0],[9.0,10.0,11.0,12.0]]]
      x2 = [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]
      x3 = [[[1.0,2.0,4.0,8.0],[1.0,2.0,4.0,8.0]],[[4.0,8.0,16.0,32.0],[4.0,8.0,16.0,32.0]],[[8.0,16.0,32.0,64.0],[8.0,16.0,32.0,64.0]]]
      result = tf.user_ops.lookaheadgrad(x1,x2,x3)
      self.assertAllEqual(result[0].eval(), [[[1.0,4.0,12.0,32.0],[1.0,4.0,12.0,32.0]],[[9.0,28.0,76.0,192.0],[9.0,28.0,76.0,192.0]],[[28.0,80.0,208.0,512.0],[28.0,80.0,208.0,512.0]]])
      self.assertAllEqual(result[1].eval(), [[186.0,424.0,952.0,2112.0],[82.0,184.0,408.0,896.0]])
      result_2 = tf.user_ops.lookaheadgrad(x1,x2,x3)
      self.assertAllEqual(result_2[0].eval(), [[[1.0,4.0,12.0,32.0],[1.0,4.0,12.0,32.0]],[[9.0,28.0,76.0,192.0],[9.0,28.0,76.0,192.0]],[[28.0,80.0,208.0,512.0],[28.0,80.0,208.0,512.0]]])
      self.assertAllEqual(result_2[1].eval(), [[186.0,424.0,952.0,2112.0],[82.0,184.0,408.0,896.0]])
      xw = tf.user_ops.lookahead(x1,x2)
      w_grad1 = tf.gradients(xw, [x1, x2])
      print(w_grad1[0], w_grad1[1])

if __name__ == '__main__':
  tf.test.main()
