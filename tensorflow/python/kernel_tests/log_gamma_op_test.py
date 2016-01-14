import tensorflow as tf

import numpy as np
import scipy.special as special

class LogGammaTest(tf.test.TestCase):
  def testLogGamma(self):
    with self.test_session():
      test_input = np.arange(1.,7.)    
      result = tf.user_ops.log_gamma(test_input)
      self.assertAllEqual(result.eval(), special.gammaln( test_input ) )
  
  #not sure if this is the correct place for this test but seems reasonable.
  def testLogGammaGradient(self):
    with self.test_session():
      raw_input = np.arange(1.,4.) 
      test_input = tf.constant( raw_input  )
      log_gamma = tf.user_ops.log_gamma( test_input )
      simple_gradient = tf.gradients( log_gamma, test_input )[0] 
      self.assertAllEqual( simple_gradient.eval(), special.psi( raw_input ) )

if __name__ == "__main__":
  tf.test.main()
