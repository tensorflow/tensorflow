import tensorflow as tf

import numpy as np
import scipy.special as special

class TriangleTest(tf.test.TestCase):
  def testTriangleLower(self):
    with self.test_session():
      rng = np.random.RandomState(1)
      test_input = rng.randn( 2, 2 )    
      result = tf.user_ops.triangle( tf.constant(test_input), 'lower' )
      self.assertAllEqual(result.eval(), np.tril( test_input ) )

  def testTriangleUpper(self):
    with self.test_session():
      rng = np.random.RandomState(1)        
      test_input = rng.randn( 2, 2 )    
      result = tf.user_ops.triangle( tf.constant(test_input), 'upper' )
      self.assertAllEqual(result.eval(), np.triu( test_input ) )

  def testTriangleGrad(self):
    with self.test_session():
      rng = np.random.RandomState(1)
      raw_input = rng.randn( 2, 2 )
      test_input = tf.constant( raw_input )
      result_a = tf.reduce_sum( tf.square( tf.user_ops.triangle(test_input,'lower') ) )
      simple_gradient_a = tf.gradients( result_a, test_input )[0]
      self.assertAllEqual( simple_gradient_a.eval(), 2.*np.tril( raw_input ) )
      result_b = tf.reduce_sum( tf.square( tf.user_ops.triangle(test_input,'upper') ) )
      simple_gradient_b = tf.gradients( result_b, test_input )[0]
      self.assertAllEqual( simple_gradient_b.eval(), 2.*np.triu( raw_input ) )      
      
if __name__ == "__main__":
  tf.test.main()
