# Tests for immediate.Env

import tensorflow as tf
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):

  def testInit(self):
    env = immediate.Env()
    self.assertTrue(True)


if __name__ == "__main__":
  tf.test.main()
