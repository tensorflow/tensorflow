from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ZeroOutTest(tf.test.TestCase):
  def test(self):
    with self.test_session():
      print(1)
      result = tf.user_ops.trial([[5, 4, 3], [3,2, 1],[1,2,3]])
      print(tf.shape(result))
      print(result.eval())
      self.assertAllEqual(result.eval(), [[5, 4, 0],[0,0,0],[0,0,0]])

if __name__ == '__main__':
  tf.test.main()
