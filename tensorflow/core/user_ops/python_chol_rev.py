import numpy as np

def chol_rev( L, L_bar ):
    N = L.shape[0]
    NB = 2
    A_bar = np.tril( L_bar )
    for Ji in range( (N-NB+1), (1-NB+1)-1, -NB ):
        J = max(1, Ji);
        JB = NB - (J - Ji); # corrected block-size        
        A_bar[ (J+JB-1):, (J-1):(J+JB-1) ] = np.linalg.solve( L[(J-1):(J+JB-1), (J-1):(J+JB-1)].T, A_bar[ (J+JB-1):, (J-1):(J+JB-1)].T ).T
        A_bar[ (J-1):(J+JB-1), (J-1):(J+JB-1)] = A_bar[(J-1):J+JB-1, (J-1):J+JB-1] - np.tril( np.dot( A_bar[(J+JB-1):, (J-1):J+JB-1].T, L[(J+JB-1):, (J-1):J+JB-1] ) )
        A_bar[ (J+JB-1):N, 0:(J-1)]  = A_bar[(J+JB-1):N, 0:J-1] - np.dot( A_bar[(J+JB-1):N, (J-1):J+JB-1], L[(J-1):J+JB-1, 0:J-1] )
        A_bar[ (J-1):(J+JB-1), 0:(J-1) ] = A_bar[ (J-1):J+JB-1, 0:J-1 ] - np.dot( A_bar[ (J+JB-1):N, (J-1):J+JB-1].T , L[(J+JB-1):N, 0:J-1] )
        A_bar[ (J-1):(J+JB-1), (J-1):(J+JB-1) ] = chol_rev_unblocked(L[(J-1):(J+JB-1), (J-1):(J+JB-1)], A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)] )
        A_bar[ (J-1):(J+JB-1), 0:(J-1) ] = A_bar[ (J-1):(J+JB-1), 0:(J-1) ] - np.dot( (A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)] + A_bar[(J-1):(J+JB-1), (J-1):(J+JB-1)].T ),L[(J-1):(J+JB-1), 0:(J-1) ] );
    return A_bar
        
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
    N = 4 
    a = np.tril( (np.array( range( 1, N**2+1, 1) )**2).reshape( N, N ).T *1. )
    b = np.tril( np.array( range( 1, N**2+1, 1) ).reshape( N, N ).T *2. )
    print "a ", a
    print "b ", b
    print "chol_rev( a , b ) ", chol_rev( a , b )
