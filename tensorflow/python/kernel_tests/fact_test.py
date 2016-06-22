from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FactTest(tf.test.TestCase):

  def test(self):
    with self.test_session():
      print(tf.user_ops.my_fact().eval())


if __name__ == '__main__':
  tf.test.main()
