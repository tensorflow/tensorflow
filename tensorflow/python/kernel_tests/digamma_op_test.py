import tensorflow as tf

import numpy as np
import scipy.special as special

class DigammaTest(tf.test.TestCase):
  def testDigamma(self):
    with self.test_session():
      test_input = np.arange(1.,4.)    
      result = tf.user_ops.digamma(test_input)
      self.assertAllEqual(result.eval(), special.psi( test_input ) )

if __name__ == "__main__":
  tf.test.main()
