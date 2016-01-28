import numpy as np

def chol_rev_unblocked( L, L_bar ):
    N = L.shape[0]
    A_bar = np.tril( L_bar )
    for J in range(N-1,-1,-1):
        A_bar[J,J] = A_bar[J,J] - np.dot(L[J+1:N,J].T,A_bar[J+1:N,J] ) / L[J,J]
        A_bar[J:N,J] = A_bar[J:N,J] / L[J,J]
        A_bar[J,0:J] = A_bar[J,0:J] - np.dot(A_bar[J:N,J].T , L[J:N,0:J] )
        A_bar[J+1:N,0:J] = A_bar[J+1:N,0:J] - np.outer(A_bar[J+1:N,J],L[J,0:J] )
        A_bar[J,J] = 0.5 * A_bar[J,J]
    return A_bar

if __name__ == "__main__":
    N = 3 
    a = np.tril( (np.array( range( 1, N**2+1, 1) )**2).reshape( N, N ).T *1. )
    b = np.tril( np.array( range( 1, N**2+1, 1) ).reshape( N, N ).T *2. )
    print "a ", a
    print "b ", b
    print "chol_rev_unblocked( a , b ) ", chol_rev_unblocked( a , b.copy() )
    print "a ", a
    print "b, ", b
