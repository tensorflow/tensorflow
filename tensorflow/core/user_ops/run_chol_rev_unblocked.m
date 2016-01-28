N = 3
a = tril( reshape( 1:(N^2), N, N ).^2 )
b = tril( reshape( 1:(N^2), N, N ) * 2 ) 
c = chol_rev_unblocked( a , b )
exit;
