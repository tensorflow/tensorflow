"""Tests for GridRNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class GridRNNCellTest(tf.test.TestCase):

  def testGridBasicRNNCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 2])
        g, _ = tf.contrib.grid_rnn.Grid1BasicRNNCell(2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g], {x.name: np.array([[1., 1.]]),
                             m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))


if __name__ == "__main__":
  tf.test.main()
