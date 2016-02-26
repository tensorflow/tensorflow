import tensorflow as tf

import numpy as np
import scipy.linalg as linalg

from scipy.linalg.blas import dsymv

#Reference Python implementation ported from Iain Murray Matlab code. Blocked version.
def chol_rev( L, L_bar, block_size=3 ):
    N = L.shape[0]
    NB = block_size
    A_bar = np.tril( L_bar )
    for Ji in range( (N-NB+1), (1-NB+1)-1, -NB ):
        J = max(1, Ji)
        JB = NB - (J - Ji) # corrected block-size      
        A_bar[ (J+JB-1):, (J-1):(J+JB-1) ] = np.linalg.solve( L[(J-1):(J+JB-1), (J-1):(J+JB-1)].T, A_bar[ (J+JB-1):, (J-1):(J+JB-1)].T ).T
        A_bar[ (J-1):(J+JB-1), (J-1):(J+JB-1)] = A_bar[(J-1):J+JB-1, (J-1):J+JB-1] - np.tril( np.dot( A_bar[(J+JB-1):, (J-1):J+JB-1].T, L[(J+JB-1):, (J-1):J+JB-1] ) )
        A_bar[ (J+JB-1):N, 0:(J-1)]  = A_bar[(J+JB-1):N, 0:J-1] - np.dot( A_bar[(J+JB-1):N, (J-1):J+JB-1], L[(J-1):J+JB-1, 0:J-1] )
        A_bar[ (J-1):(J+JB-1), 0:(J-1) ] = A_bar[ (J-1):J+JB-1, 0:J-1 ] - np.dot( A_bar[ (J+JB-1):N, (J-1):J+JB-1].T , L[(J+JB-1):N, 0:J-1] )
        A_bar[ (J-1):(J+JB-1), (J-1):(J+JB-1) ] = chol_rev_unblocked(L[(J-1):(J+JB-1), (J-1):(J+JB-1)], A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)] )
        A_bar[ (J-1):(J+JB-1), 0:(J-1) ] = A_bar[ (J-1):(J+JB-1), 0:(J-1) ] - np.dot( (A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)] + A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)].T ),L[(J-1):(J+JB-1), 0:(J-1) ] );
    return A_bar

#Reference Python implementation ported from Iain Murray Matlab code. Unblocked version.
def chol_rev_unblocked( L, L_bar ):
    N = L.shape[0]
    A_bar = np.tril( L_bar )
    for J in range(N-1,-1,-1):
        A_bar[J,J] = A_bar[J,J] - np.dot(L[J+1:N,J].T,A_bar[J+1:N,J] ) / L[J,J]
        A_bar[J:N,J] = A_bar[J:N,J] / L[J,J]
        A_bar[J,0:J] = A_bar[J,0:J] - np.dot(A_bar[J:N,J].T , L[J:N,0:J] )
        A_bar[J+1:N,0:J] = A_bar[J+1:N,0:J] - np.outer(A_bar[J+1:N,J],L[J,0:J] )
        A_bar[J,J] = 0.5 * A_bar[J,J];
    return A_bar

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

class CholeskyReferenceTest(tf.test.TestCase):
  def testCholeskyReferenceA(self):
    nDim = 8      
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.rand(nDim,nDim) )
    raw_g = np.tril( rng.rand(nDim,nDim) )      
    with self.test_session():
      murray_blocked = chol_rev( raw_a, raw_g)
      murray_unblocked = chol_rev_unblocked( raw_a, raw_g)
      hips = cholesky_grad_python( raw_a, raw_g )
      self.assertAllClose( murray_blocked, murray_unblocked )
      self.assertAllClose( 0.5*(murray_unblocked+murray_unblocked.T) , hips )

  def testCholeskyReferenceB(self):
    nDim = 30      
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.rand(nDim,nDim) )
    raw_g = np.tril( rng.rand(nDim,nDim) )      
    with self.test_session():
      murray_blocked = chol_rev( raw_a, raw_g)
      murray_unblocked = chol_rev_unblocked( raw_a, raw_g)
      hips = cholesky_grad_python( raw_a, raw_g )
      self.assertAllClose( murray_blocked, murray_unblocked )
      self.assertAllClose( 0.5*(murray_unblocked+murray_unblocked.T) , hips )

  def testCholeskyReferenceC(self):
    nDim = 37      
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.rand(nDim,nDim) )
    raw_g = np.tril( rng.rand(nDim,nDim) )      
    with self.test_session():
      murray_blocked = chol_rev( raw_a, raw_g, block_size = 7)
      murray_unblocked = chol_rev_unblocked( raw_a, raw_g)
      hips = cholesky_grad_python( raw_a, raw_g )
      self.assertAllClose( murray_blocked, murray_unblocked )
      self.assertAllClose( 0.5*(murray_unblocked+murray_unblocked.T) , hips )

  def testCholeskyReferenceD(self):
    nDim = 30      
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.rand(nDim,nDim) )
    raw_g = np.tril( rng.rand(nDim,nDim) )      
    with self.test_session():
      murray_blocked = chol_rev( raw_a, raw_g, block_size = 32)
      murray_unblocked = chol_rev_unblocked( raw_a, raw_g)
      hips = cholesky_grad_python( raw_a, raw_g )
      self.assertAllClose( murray_blocked, murray_unblocked )
      self.assertAllClose( 0.5*(murray_unblocked+murray_unblocked.T) , hips )

  def testCholeskyReferenceE(self):
    nDim = 30    
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.randn(nDim,nDim) )
    raw_g = np.tril( rng.randn(nDim,nDim) )
    with self.test_session():
      a = tf.constant( raw_a )
      g = tf.constant( raw_g )
      temp = chol_rev_unblocked( raw_a, raw_g ) 
      test = cholesky_grad_python( raw_a, raw_g )
      reference = 0.5*( temp + temp.T )
      self.assertTrue( (np.abs( test - reference) < 1e-4).all().all() )
      
class CholeskyGradTest(tf.test.TestCase):
  def testCholeskyGrad(self):
    nDim = 4      
    rng = np.random.RandomState(1)
    raw_b = np.eye(nDim)*3
    raw_y = np.tril( rng.randn(nDim,nDim) )  
    with self.test_session():
      b = tf.constant( raw_b )
      a = tf.cholesky( b )
      y = tf.constant( raw_y )
      intermediate = tf.user_ops.triangular_solve(a,y,'lower')
      linear_solution = tf.user_ops.triangular_solve(tf.transpose(a),intermediate,'upper') 
      x = tf.reduce_sum( linear_solution )
      grad_b = tf.gradients( x, b )[0]
      r= linalg.solve( raw_b, np.ones( nDim ) )      
      S = linalg.solve( raw_b, np.dot( raw_y, np.ones(nDim) ) )
      referenceValue = -1*np.outer( r, S )
      testValue = grad_b.eval()
      testValue = 0.5 * ( testValue + testValue.T )
      referenceValue = 0.5*(referenceValue + referenceValue.T  )
      self.assertAllClose( testValue , referenceValue )

  def testCholeskyGradB(self):
    nDim = 10    
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.randn(nDim,nDim) )
    raw_g = np.tril( rng.randn(nDim,nDim) )
    with self.test_session():
      a = tf.constant( raw_a )
      g = tf.constant( raw_g )
      test = tf.user_ops.cholesky_grad( a, g, 'lower' ).eval()
      temp = chol_rev( raw_a, raw_g ) 
      reference = 0.5*( temp + temp.T )
      self.assertTrue( (np.abs( test - reference) < 1e-4).all().all() )

  def testCholeskyGradC(self):
    nDim = 30    
    rng = np.random.RandomState(1)
    raw_a = np.tril( rng.randn(nDim,nDim) )
    raw_g = np.tril( rng.randn(nDim,nDim) )
    with self.test_session():
      a = tf.constant( raw_a )
      g = tf.constant( raw_g )
      test = tf.user_ops.cholesky_grad( a, g, 'lower' ).eval()
      temp = chol_rev( raw_a, raw_g, block_size=32 ) 
      reference = 0.5*( temp + temp.T )
      #print "np.abs( test - reference)  ", np.abs( test - reference) 
      self.assertTrue( (np.abs( test - reference) < 1e-4).all().all() )

  def testCholeskyGradD(self):
    rng = np.random.RandomState(1)    
    a = rng.randn( 5, 3 )
    with self.test_session():
      raw_b = np.array( np.dot( a.T, a )  )
      b = tf.constant( raw_b )
      a = tf.cholesky( b )
      diagonal = tf.pack([a[i,i] for i in range(3)])
      logDeterminant = tf.reduce_sum( tf.log( tf.square( diagonal  ) ) )
      test_value = tf.gradients( logDeterminant, b )[0].eval()
      referenceValue = np.linalg.inv(raw_b)
      self.assertAllClose( test_value, referenceValue  )
 

if __name__ == "__main__":
  tf.test.main()
