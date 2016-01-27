/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("MatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the determinant of a square matrix.

input: A tensor of shape `[M, M]`.
output: A scalar, equal to the determinant of the input.
)doc");

REGISTER_OP("BatchMatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the determinants for a batch of square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a 1-D tensor containing the determinants
for all input submatrices `[..., :, :]`.

input: Shape is `[..., M, M]`.
output: Shape is `[...]`.
)doc");

REGISTER_OP("MatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the inverse of a square invertible matrix.

The op uses the Cholesky decomposition if the matrix is symmetric positive
definite and LU decomposition with partial pivoting otherwise.

If the matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

input: Shape is `[M, M]`.
output: Shape is `[M, M]` containing the matrix inverse of the input.
)doc");

REGISTER_OP("BatchMatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the inverse of square invertible matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses the Cholesky decomposition if the matrices are symmetric positive
definite and LU decomposition with partial pivoting otherwise.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M, M]`.
)doc");

REGISTER_OP("Cholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Doc(R"doc(
Calculates the Cholesky decomposition of a square matrix.

The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The result is the lower-triangular matrix of the Cholesky decomposition of the
input.

input: Shape is `[M, M]`.
output: Shape is `[M, M]`.
)doc");

REGISTER_OP("BatchCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Doc(R"doc(
Calculates the Cholesky decomposition of a batch of square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M, M]`.
)doc");

REGISTER_OP("SelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Doc(R"doc(
Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

Only the lower-triangular part of the input will be used in this case. The
upper-triangular part will not be read.

The result is a M+1 x M matrix whose first row is the eigenvalues, and
subsequent rows are eigenvectors.

input: Shape is `[M, M]`.
output: Shape is `[M+1, M]`.
)doc");

REGISTER_OP("BatchSelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Doc(R"doc(
Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M+1, M]`.
)doc");

REGISTER_OP("MatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Solves a system of linear equations. Checks for invertibility.

matrix: Shape is `[M, M]`.
rhs: Shape is `[M, K]`.
output: Shape is `[M, K]` containing the tensor that solves
matrix * output = rhs.
)doc");

REGISTER_OP("BatchMatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Solves systems of linear equations. Checks for invertibility.

Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape
`[..., M, K]`. The output is a tensor shape `[..., M, K]` where each output
matrix satisfies matrix[..., :, :] * output[..., :, :] = rhs[..., :, :].

matrix: Shape is `[..., M, M]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., M, K]`.
)doc");

REGISTER_OP("MatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Solves a system of linear equations with an upper or lower triangular matrix by
backsubstitution.

`matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
upper triangular part of `matrix` is ignored. If `lower` is False then the
strictly lower triangular part of `matrix` is ignored. `rhs` is a matrix of
shape [M, K]`.

The output is a matrix of shape `[M, K]`. If `lower` is `True` then the output
satisfies \\(\sum_{k=0}^{i}\\) matrix[i, k] * output[k, j] = rhs[i, j].
If `lower` is false then output satisfies
\\(\sum_{k=i}^{K-1}\\) matrix[i, k] * output[k, j] = rhs[i, j].

matrix: Shape is `[M, M]`.
rhs: Shape is `[M, K]`.
output: Shape is `[M, K]`.
lower: Boolean indicating whether matrix is lower or upper triangular.
)doc");

REGISTER_OP("BatchMatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Solves systems of linear equations with upper or lower triangular matrices by
backsubstitution.

`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is ignored. If `lower` is False then the strictly
lower triangular part of each inner-most matrix is ignored. `rhs` is a tensor
of shape [..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `lower` is `True` then the
output satisfies
\\(\sum_{k=0}^{i}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j].
If `lower` is false then the strictly then the output satisfies
\\(sum_{k=i}^{K-1}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j].

matrix: Shape is `[..., M, M]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., M, K]`.
lower: Boolean indicating whether matrix is lower or upper triangular.
)doc");

REGISTER_OP("MatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("fast: bool = True")
    .Doc(R"doc(
Solves a linear least-squares problem.

Below we will use the following notation
`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
\lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
\\(X = A^T (A A^T + \lambda I)^{-1} B\\),
which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
subject to \\(A Z = B\\).
Notice that the fast path is only numerically stable when \\(A\\) is
numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
or \\(\lambda\\) is sufficiently large.

If `fast` is `False` then the solution is computed using the rank revealing QR
decomposition with column pivoting. This will always compute a least-squares
solution that minimizes the residual norm \\(||A X - B||_F^2 \\), even when
\\( A \\) is rank deficient or ill-conditioned. Notice: The current version
does not compute a minimum norm solution. If `fast` is `False` then
`l2_regularizer` is ignored.

matrix: Shape is `[M, N]`.
rhs: Shape is `[M, K]`.
output: Shape is `[N, K]` containing the tensor that solves
  `matrix * output = rhs` in the least-squares sense.
)doc");

REGISTER_OP("BatchMatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("fast: bool = True")
    .Doc(R"doc(
Solves multiple linear least-squares problems.

`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape `[..., M, K]`. The output
is a tensor shape `[..., N, K]` where each output matrix solves each of
the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :] in the
least squares sense.

Below we will use the following notation for each pair of
matrix and right-hand sides in the batch:

`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
\lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
\\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
minimum-norm solution to the under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
\\(A Z = B\\). Notice that the fast path is only numerically stable when
\\(A\\) is numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\) is
sufficiently large.

If `fast` is `False` then the solution is computed using the rank revealing QR
decomposition with column pivoting. This will always compute a least-squares
solution that minimizes the residual norm \\(||A X - B||_F^2\\), even when
\\(A\\) is rank deficient or ill-conditioned. Notice: The current version does
not compute a minimum norm solution. If `fast` is `False` then `l2_regularizer`
is ignored.

matrix: Shape is `[..., M, N]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., N, K]`.
)doc");

}  // namespace tensorflow
