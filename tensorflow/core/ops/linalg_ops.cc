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
T: The type of values in the input and output.
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
T: The type of values in the input and output.
)doc");

REGISTER_OP("MatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the inverse of a square invertible matrix. Checks for invertibility.

input: Shape is `[M, M]`.
output: Shape is `[M, M]` containing the matrix inverse of the input.
T: The type of values in the input and output.
)doc");

REGISTER_OP("BatchMatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Calculates the inverse of square invertible matrices. Checks for invertibility.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M, M]`.
T: The type of values in the input and output.
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
T: The type of values in the input and output.
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
T: The type of values in the input and output.
)doc");

}  // namespace tensorflow
