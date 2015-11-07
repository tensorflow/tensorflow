# pylint: disable=g-bad-import-order,unused-import
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import linear


class LinearTest(tf.test.TestCase):

  def testLinear(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(1.0)):
        x = tf.zeros([1, 2])
        l = linear.linear([x], 2, False)
        sess.run([tf.variables.initialize_all_variables()])
        res = sess.run([l], {x.name: np.array([[1., 2.]])})
        self.assertAllClose(res[0], [[3.0, 3.0]])

        # Checks prevent you from accidentally creating a shared function.
        with self.assertRaises(ValueError) as exc:
          l1 = linear.linear([x], 2, False)
        self.assertEqual(exc.exception.message[:12], "Over-sharing")

        # But you can create a new one in a new scope and share the variables.
        with tf.variable_scope("l1") as new_scope:
          l1 = linear.linear([x], 2, False)
        with tf.variable_scope(new_scope, reuse=True):
          linear.linear([l1], 2, False)
        self.assertEqual(len(tf.trainable_variables()), 2)


if __name__ == "__main__":
  tf.test.main()
