import tensorflow as tf

import numpy as np
import scipy.linalg as linalg

from scipy.linalg.blas import dsymv

#Reference Python implementation taken from HIPS autograd which was originally based on Sheffield GPy.
def cholesky_grad_python(L,g):
	N = L.shape[0]
	dL = np.tril(g)
        dL[-1,-1] /= 2 * L[-1,-1]
        for k in range(N-2, -1, -1):
            dL[k+1:,k] -= dsymv(1., dL[k+1:,k+1:], L[k+1:,k], lower=True)
            dL[k+1:,k] -= np.diag(dL[k+1:,k+1:]) * L[k+1:,k]
            dL[k+1:,k] /= L[k,k]
            dL[k,k] -= np.dot(dL[k+1:,k], L[k+1:,k])
            dL[k,k] /= 2 * L[k,k]
        return (dL + dL.T)/2.

class CholeskyGradTest(tf.test.TestCase):
  def testCholeskyGrad(self):
    with self.test_session():
      raw_b = np.array( [[2.,0.],[0.,2.]]  )
      b = tf.constant( raw_b )
      a = tf.cholesky( b )
      raw_y = np.array( [[1.,2.],[3.,4.]] )
      y = tf.constant( raw_y )
      intermediate = tf.user_ops.triangular_solve(a,y,'lower')
      linear_solution = tf.user_ops.triangular_solve(tf.transpose(a),intermediate,'upper') 
      x = tf.reduce_sum( linear_solution )
      grad_b = tf.gradients( x, b )[0]
      r= linalg.solve( raw_b, np.ones( 2 ) )
      S = linalg.solve( raw_b, np.dot( raw_y, np.ones(2) ) )
      referenceValue = -1*np.outer( r, S )
      testValue = grad_b.eval()
      referenceValue = 0.5*(referenceValue + referenceValue.T  )
      print "referenceValue ", referenceValue
      print "testValue ", testValue	
      #self.assertAllClose( testValue , referenceValue )
      
  def testCholeskyGradB(self):
    with self.test_session():
      raw_a = np.array( [[1.,0.],[1.,1.]]  )
      a = tf.constant( raw_a )
      raw_g = np.array( [[1.,2.],[3.,4.]] )
      g = tf.constant( raw_g )
      cg = tf.user_ops.cholesky_grad( a, g, 'lower' )
      self.assertAllEqual( cg.eval(), cholesky_grad_python( raw_a, raw_g )  )

  def testCholeskyGradC(self):
    with self.test_session():
      raw_b = np.array( [[1.5,1.],[1.,1.5]]  )
      b = tf.constant( raw_b )
      a = tf.cholesky( b )
      diagonal = tf.pack([a[i,i] for i in range(2)])
      logDeterminant = tf.reduce_sum( tf.log( tf.square( diagonal  ) ) )
      grad_b = tf.gradients( logDeterminant, b )[0]
      referenceValue = np.linalg.inv(raw_b)
      self.assertAllClose( grad_b.eval(), referenceValue  )

if __name__ == "__main__":
  tf.test.main()
