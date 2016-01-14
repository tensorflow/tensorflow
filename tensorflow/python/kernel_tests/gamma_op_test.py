import tensorflow as tf

import numpy as np
import scipy.special as special

class GammaTest(tf.test.TestCase):
  def testGamma(self):
    with self.test_session():
      test_input = np.arange(1.,7.)    
      result = tf.user_ops.gamma(test_input)
      self.assertAllEqual(result.eval(), special.gamma( test_input ) )

  def testGammaGradient(self):
    with self.test_session():
      raw_input = np.arange(1.,4.) 
      test_input = tf.constant( raw_input  )
      gamma = tf.user_ops.gamma( test_input )
      simple_gradient = tf.gradients( gamma, test_input )[0] 
      self.assertAllEqual(simple_gradient.eval(), special.psi( raw_input )* special.gamma( raw_input ) )

if __name__ == "__main__":
  tf.test.main()
