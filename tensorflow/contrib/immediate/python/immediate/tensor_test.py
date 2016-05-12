# Tests for immediate.Tensor

import tensorflow as tf
import tensorflow.contrib.immediate as immediate


class TensorTest(tf.test.TestCase):

  def testInit(self):
    tensor = immediate.Tensor(None, None)
    self.assertTrue(True)


if __name__ == "__main__":
  tf.test.main()
