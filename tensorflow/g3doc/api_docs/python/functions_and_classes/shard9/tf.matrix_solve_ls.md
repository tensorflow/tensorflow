### `tf.matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)` {#matrix_solve_ls}

Solves a linear least-squares problem.

Below we will use the following notation
`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the regularized
least-squares problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}}
||A Z - B||_F^2 + \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is
computed as \\(X = A^T (A A^T + \lambda I)^{-1} B\\),
which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
subject to \\(A Z = B\\).
Notice that the fast path is only numerically stable when \\(A\\) is
numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
or \\(\lambda\\) is sufficiently large.

If `fast` is `False` then the solution is computed using the rank revealing
QR decomposition with column pivoting. This will always compute a
least-squares solution that minimizes the residual norm
\\(||A X - B||_F^2 \\), even when \\(A\\) is rank deficient or
ill-conditioned. Notice: The current version does not compute a minimum norm
solution. If `fast` is `False` then `l2_regularizer` is ignored.

##### Args:


*  <b>`matrix`</b>: 2-D `Tensor` of shape `[M, N]`.
*  <b>`rhs`</b>: 2-D `Tensor` of shape is `[M, K]`.
*  <b>`l2_regularizer`</b>: 0-D  `double` `Tensor`. Ignored if `fast=False`.
*  <b>`fast`</b>: bool. Defaults to `True`.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`output`</b>: Matrix of shape `[N, K]` containing the matrix that solves
    `matrix * output = rhs` in the least-squares sense.

