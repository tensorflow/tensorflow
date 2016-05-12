# Tests for immediate.Op

import tensorflow as tf
import tensorflow.contrib.immediate as immediate


class OpTest(tf.test.TestCase):

  def testInit(self):
    op = immediate.Op()
    self.assertTrue(True)


if __name__ == "__main__":
  tf.test.main()
