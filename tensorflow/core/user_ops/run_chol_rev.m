N = 4
a = tril( reshape( 1:(N^2), N, N ).^2 )
b = tril( reshape( 1:(N^2), N, N ) * 2 ) 
c = chol_rev( a , b )
exit;
