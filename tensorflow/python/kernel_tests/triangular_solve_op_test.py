import tensorflow as tf

import numpy as np
import scipy.linalg as linalg

class SolveTest(tf.test.TestCase):
  def testSolve(self):
    with self.test_session():
      raw_a = np.array( [[1.,0.,0.], [1.,2.,0.], [1.,2.,3.]] )     
      a_lower = tf.constant( raw_a )
      a_upper = tf.constant( raw_a.T )
      raw_y = np.array( [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.]])
      y = tf.constant( raw_y )
      y_t = tf.constant( raw_y.T )
      x_lower = tf.user_ops.triangular_solve(a_lower,y,'lower')
      x_upper = tf.user_ops.triangular_solve(a_upper,y,'upper')
      reference_result_lower = linalg.solve_triangular( raw_a , raw_y,  lower=True )
      reference_result_upper = linalg.solve_triangular( raw_a.T , raw_y,  lower=False )
      self.assertAllEqual(x_lower.eval(), reference_result_lower )
      self.assertAllEqual(x_upper.eval(), reference_result_upper )
      with self.assertRaises(tf.errors.InvalidArgumentError):
        x_fail = tf.user_ops.triangular_solve(a_upper,y_t,'upper')
        x_fail.eval()
      vector = tf.constant( np.array( [1., 2., 3. ] ) )        
      with self.assertRaises(tf.errors.InvalidArgumentError):      
        x_fail_B = tf.user_ops.triangular_solve(a_upper,vector,'lower')
        x_fail_B.eval()    

  def testSolveGrad(self):
    with self.test_session():
      raw_a = np.array( [[2.,0.],[1.,2.]]  )
      a = tf.constant( raw_a )
      raw_y = np.array( [[1.,2.],[3.,4.]] )
      y = tf.constant( raw_y )
      x = tf.reduce_sum( tf.user_ops.triangular_solve(a,y,'lower') )
      grad_y = tf.gradients( x, y )[0]
      grad_a = tf.gradients( x, a )[0]
      reference_grad = np.outer( np.linalg.inv(raw_a).sum(axis=0), np.ones(2) )
      r= linalg.solve( raw_a.T, np.ones( 2 ) )
      S = linalg.solve( raw_a, np.dot( raw_y, np.ones(2) ) )
      self.assertAllEqual( grad_y.eval(), reference_grad )
      self.assertAllEqual( grad_a.eval(), -1*np.outer( r, S ) )
      

if __name__ == "__main__":
  tf.test.main()
