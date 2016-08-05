/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

namespace {

// Return in <out> the result of making <s> a square matrix.
Status MakeSquareMatrix(InferenceContext* c, const Shape* s,
                        const Shape** out) {
  TF_RETURN_IF_ERROR(c->WithRank(s, 2, &s));
  const Dimension* d;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, 0), c->Dim(s, 1), &d));
  *out = c->Matrix(d, d);
  return Status::OK();
}

Status UnchangedSquareShapeFn(InferenceContext* c) {
  const Shape* out;
  TF_RETURN_IF_ERROR(MakeSquareMatrix(c, c->input(0), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Return in <out> the result of making the end of <s> a square matrix.
Status MakeBatchSquareMatrix(InferenceContext* c, const Shape* input,
                             const Shape** out) {
  const Shape* s;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &s));

  const Dimension* d;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, -2), c->Dim(s, -1), &d));

  const Shape* batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(s, 0, -2, &batch_shape));
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(d, d), out));
  return Status::OK();
}

Status BatchUnchangedSquareShapeFn(InferenceContext* c) {
  const Shape* out;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &out));
  c->set_output(0, out);
  return Status::OK();
}

Status SquareMatrixSolveShapeFn(InferenceContext* c) {
  const Shape* lhs;
  const Shape* rhs;
  TF_RETURN_IF_ERROR(MakeSquareMatrix(c, c->input(0), &lhs));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &rhs));

  // lhs and rhs have the same number of rows. Make a new output
  // shape that uses rows to replace rhs.dim[0].
  const Dimension* rows;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, 0), c->Dim(rhs, 0), &rows));
  const Shape* out;
  TF_RETURN_IF_ERROR(c->ReplaceDim(rhs, 0, rows, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Inputs are [...,M,N] and [...,M,K].  Output is [...,N,K].
// If <square>, then input is [...,M,M].
Status BatchMatrixSolveShapeFn(InferenceContext* c, bool square) {
  const Shape* lhs;
  const Shape* rhs;
  if (square) {
    TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  }
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  // Make the common batch subshape between the two dimensions.
  const Shape* lhs_batch_shape;
  const Shape* batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &batch_shape));
  TF_RETURN_IF_ERROR(c->Merge(lhs_batch_shape, batch_shape, &batch_shape));

  // lhs and rhs have the same value for m.
  const Dimension* m;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, -2), c->Dim(rhs, -2), &m));

  const Dimension* n = c->Dim(lhs, -1);
  if (square) {
    TF_RETURN_IF_ERROR(c->Merge(m, n, &n));
  }

  // Build final shape (batch_shape + n + k) in <out>.
  const Shape* out;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(n), &out));
  TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(c->Dim(rhs, -1)), &out));
  c->set_output(0, out);
  return Status::OK();
}

Status BatchSvdShapeHelperFn(InferenceContext* c, const Shape* input) {
  const Dimension* m = c->Dim(input, -2);
  const Dimension* n = c->Dim(input, -1);
  const Dimension* p;
  TF_RETURN_IF_ERROR(c->Min(m, n, &p));
  const Shape* batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  const Shape* e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(p), &e_shape));
  c->set_output(0, e_shape);
  bool compute_uv;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_uv", &compute_uv));
  if (compute_uv) {
    const Shape* u_shape;
    const Shape* v_shape;
    bool full_matrices;
    TF_RETURN_IF_ERROR(c->GetAttr("full_matrices", &full_matrices));
    if (full_matrices) {
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(m, m), &u_shape));
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(n, n), &v_shape));
    } else {
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(m, p), &u_shape));
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(n, p), &v_shape));
    }
    c->set_output(1, u_shape);
    c->set_output(2, v_shape);
  } else {
    c->set_output(1, c->Vector(0ll));
    c->set_output(2, c->Vector(0ll));
  }
  return Status::OK();
}

// Input is [M,N].  First output is [min(M,N)].
// Second and third outputs are:
//   [0]; [0], if compute_uv is false.
//   [M,M]; [N,N], if compute_uv is true and full_matrices is true,
//   [M,P]; [N,P], if compute_uv is true and full_matrices is false,
// where P = min(M,N).
Status SvdShapeFn(InferenceContext* c) {
  const Shape* input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  return BatchSvdShapeHelperFn(c, input);
}

// Input is [...,M,N].  First output is [...,min(M,N)].
// Second and third outputs are:
//   [0]; [0], if compute_uv is false.
//   [...,M,M]; [...,N,N], if compute_uv is true and full_matrices is true,
//   [...,M,P]; [...,N,P], if compute_uv is true and full_matrices is false,
// where P = min(M,N).
Status BatchSvdShapeFn(InferenceContext* c) {
  const Shape* input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
  return BatchSvdShapeHelperFn(c, input);
}

// Input is [N,N]. Outputs are:
//   [N];[0], if compute_v is false,
//   [N];[N,N], if compute_v is true.
Status SelfAdjointEigV2ShapeFn(InferenceContext* c) {
  const Shape* input;
  TF_RETURN_IF_ERROR(MakeSquareMatrix(c, c->input(0), &input));
  const Dimension* n;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, 0), c->Dim(input, 1), &n));
  c->set_output(0, c->Vector(n));
  bool compute_v;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_v", &compute_v));
  if (compute_v) {
    c->set_output(1, c->Matrix(n, n));
  } else {
    c->set_output(1, c->Vector(0ll));
  }
  return Status::OK();
}

// Input is [...,N,N]. Outputs are:
//   [...,N];[0], if compute_v is false,
//   [...,N];[...,N,N], if compute_v is true.
Status BatchSelfAdjointEigV2ShapeFn(InferenceContext* c) {
  const Shape* input;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));
  const Dimension* n;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, -2), c->Dim(input, -1), &n));
  const Shape* batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  const Shape* e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(n), &e_shape));
  c->set_output(0, e_shape);
  bool compute_v;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_v", &compute_v));
  if (compute_v) {
    const Shape* v_shape;
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(n, n), &v_shape));
    c->set_output(1, v_shape);
  } else {
    c->set_output(1, c->Vector(0ll));
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("MatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      TF_RETURN_IF_ERROR(MakeSquareMatrix(c, c->input(0), &input));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Computes the determinant of a square matrix.

input: A tensor of shape `[M, M]`.
output: A scalar, equal to the determinant of the input.
)doc");

REGISTER_OP("BatchMatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

      const Dimension* unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(input, -2), &unused));

      const Shape* out;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the determinants for a batch of square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`.

input: Shape is `[..., M, M]`.
output: Shape is `[...]`.
)doc");

REGISTER_OP("MatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(UnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the inverse of a square invertible matrix or its adjoint (conjugate
transpose).

The op uses LU decomposition with partial pivoting to compute the inverse.

If the matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

input: Shape is `[M, M]`.
output: Shape is `[M, M]`. If `adjoint` is `False` then `output` contains the
matrix inverse of `input`. If `adjoint` is `True` then `output` contains the
matrix inverse of the adjoint of `input`.
)doc");

REGISTER_OP("BatchMatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(BatchUnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the inverse of square invertible matrices or their adjoints
(conjugate transposes).

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

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
    .SetShapeFn(UnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the Cholesky decomposition of a square matrix.

The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The result is the lower-triangular matrix of the Cholesky decomposition of the
input, `L`, so that `input = L L^*`.

input: Shape is `[M, M]`.
output: Shape is `[M, M]`.
)doc");

REGISTER_OP("BatchCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .SetShapeFn(BatchUnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the Cholesky decomposition of a batch of square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M, M]`.
)doc");

REGISTER_OP("CholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(UnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527.

l: Output of Cholesky algorithm l = chol(A). Shape is `[M, M]`.
  Algorithm depends only on lower triangular part of this matrix.
grad: df/dl where f is some scalar function. Shape is `[M, M]`.
  Algorithm depends only on lower triangular part of this matrix.
output: Symmetrized version of df/dA . Shape is `[M, M]`.
)doc");

REGISTER_OP("BatchCholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BatchUnchangedSquareShapeFn)
    .Doc(R"doc(
Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527.

l: Output of batch Cholesky algorithm l = batch_cholesky(A). Shape is `[..., M, M]`.
  Algorithm depends only on lower triangular part of the innermost matrices of
  this tensor.
grad: df/dl where f is some scalar function. Shape is `[..., M, M]`.
  Algorithm depends only on lower triangular part of the innermost matrices of
  this tensor.
output: Symmetrized version of df/dA . Shape is `[..., M, M]`
)doc");

REGISTER_OP("SelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(11, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      TF_RETURN_IF_ERROR(MakeSquareMatrix(c, c->input(0), &input));

      const Dimension* d = c->Dim(input, 0);
      const Dimension* d_plus_1;
      TF_RETURN_IF_ERROR(c->Add(d, 1, &d_plus_1));
      c->set_output(0, c->Matrix(d_plus_1, d));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the Eigen Decomposition of a square Self-Adjoint matrix.

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
    .Deprecated(11, "Use BatchSelfAdjointEigV2 instead.")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));

      const Dimension* d = c->Dim(input, -1);
      const Dimension* d_plus_1;
      TF_RETURN_IF_ERROR(c->Add(d, 1, &d_plus_1));

      const Shape* s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &s));
      TF_RETURN_IF_ERROR(c->Concatenate(s, c->Matrix(d_plus_1, d), &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a [..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

input: Shape is `[..., M, M]`.
output: Shape is `[..., M+1, M]`.
)doc");

REGISTER_OP("SelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float}")
    .SetShapeFn(SelfAdjointEigV2ShapeFn)
    .Doc(R"doc(
Computes the eigen decomposition of a self-adjoint (\"symmetric\") matrix.

Computes the eigenvalues and (optionally) eigenvectors such that
`input = v * diag(e)`.

```prettyprint
# a is a self-adjoint matrix.
# e is a vector of eigenvalues.
# v is a matrix of eigenvectors.
e, v = self_adjoint_eig(a)
e = self_adjoint_eig(a, compute_v=False)
```

input: `Tensor` input of shape `[N, N]`.
compute_v: If `True` then eigenvectors will be computed and returned in `v`.
  Otherwise, only the eigenvalues will be computed.
e: Eigenvalues. Shape is `[N]`.
v: Eigenvectors. Shape is `[N, N]`.
)doc");

REGISTER_OP("BatchSelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float}")
    .SetShapeFn(BatchSelfAdjointEigV2ShapeFn)
    .Doc(R"doc(
Computes the eigen decomposition of a batch of square self-adjoint matrices.

Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`.

```prettyprint
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = batch_self_adjoint_eig(a)
e = batch_self_adjoint_eig(a, compute_v=False)
```

input: `Tensor` input of shape `[N, N]`.
compute_v: If `True` then eigenvectors will be computed and returned in `v`.
  Otherwise, only the eigenvalues will be computed.
e: Eigenvalues. Shape is `[N]`.
v: Eigenvectors. Shape is `[N, N]`.
)doc");

REGISTER_OP("MatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(SquareMatrixSolveShapeFn)
    .Doc(R"doc(
Solves a system of linear equations. Checks for invertibility.

matrix: Shape is `[M, M]`.
rhs: Shape is `[M, K]`.
output: Shape is `[M, K]`. If `adjoint` is `False` then `output` that solves
`matrix` * `output` = `rhs`. If `adjoint` is `True` then `output` that solves
`adjoint(matrix)` * `output` = `rhs`.
adjoint: Boolean indicating whether to solve with `matrix` or its adjoint.
)doc");

REGISTER_OP("BatchMatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn([](InferenceContext* c) {
      return BatchMatrixSolveShapeFn(c, true /* square (*/);
    })
    .Doc(R"doc(
Solves systems of linear equations. Checks for invertibility.

Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape
`[..., M, K]`. The output is a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output
matrix satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output
matrix satisfies `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

matrix: Shape is `[..., M, M]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., M, K]`.
adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
         adjoint.
)doc");

REGISTER_OP("MatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(SquareMatrixSolveShapeFn)
    .Doc(R"doc(
Solves a system of linear equations with an upper or lower triangular matrix by
backsubstitution.

`matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
upper triangular part of `matrix` is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of `matrix` is
assumed to be zero and not accessed.
`rhs` is a matrix of shape [M, K]`.

The output is a matrix of shape `[M, K]`. If `adjoint` is `False` the output
satisfies the matrix equation `matrix` * `output` = `rhs`.
If `adjoint` is `False` then `output` satisfies the matrix equation
`matrix` * `output` = `rhs`.
If `adjoint` is `True` then `output` satisfies the matrix equation
`adjoint(matrix)` * `output` = `rhs`.

matrix: Shape is `[M, M]`.
rhs: Shape is `[M, K]`.
output: Shape is `[M, K]`.
lower: Boolean indicating whether `matrix` is lower or upper triangular
adjoint: Boolean indicating whether to solve with `matrix` or its adjoint.
)doc");

REGISTER_OP("BatchMatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn([](InferenceContext* c) {
      return BatchMatrixSolveShapeFn(c, true /* square (*/);
    })
    .Doc(R"doc(
Solves systems of linear equations with upper or lower triangular matrices by
backsubstitution.

`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of each inner-most
matrix is assumed to be zero and not accessed.
`rhs` is a tensor of shape [..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `adjoint` is `True` then the
innermost matrices in output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

matrix: Shape is `[..., M, M]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., M, K]`.
lower: Boolean indicating whether the innermost matrices in `matrix` are
       lower or upper triangular.
adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
         adjoint.
)doc");

REGISTER_OP("MatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Attr("fast: bool = True")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* lhs;
      const Shape* rhs;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &lhs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &rhs));

      // The matrix and right-hand side must have the same number of rows.
      const Dimension* unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, 0), c->Dim(rhs, 0), &unused));

      c->set_output(0, c->Matrix(c->Dim(lhs, 1), c->Dim(rhs, 1)));
      return Status::OK();
    })
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

If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
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
    .Attr("T: {double, float}")
    .Attr("fast: bool = True")
    .SetShapeFn([](InferenceContext* c) {
      return BatchMatrixSolveShapeFn(c, false /* square */);
    })
    .Doc(R"doc(
Solves multiple linear least-squares problems.

`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form matrices of size `[M, N]`. Rhs is a tensor of shape `[..., M, K]`.
The output is a tensor shape `[..., N, K]` where each output matrix solves
each of the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]
in the least squares sense.

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

If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
`l2_regularizer` is ignored.

matrix: Shape is `[..., M, N]`.
rhs: Shape is `[..., M, K]`.
output: Shape is `[..., N, K]`.
)doc");

REGISTER_OP("Svd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(SvdShapeFn)
    .Doc(R"doc(
Computes the singular value decomposition of a matrix.

Computes the SVD of if `input` such that `input = u * diag(s) * transpose(v)`

```prettyprint
# a is a matrix.
# s is a vector of singular values.
# u is the matrix of left singular vectors.
# v is a matrix of right singular vectors.
s, u, v = svd(a)
s, _, _ = svd(a, compute_uv=False)
```

input: Shape is `[M, N]`. Let `P` be the minimum of `M` and `N`.
s: Singular values. Shape is `[P]`.
u: Left singular vectors; if `full_matrices` is `False` then shape is `[M, M]`.
  If `full_matrices` is `True` then shape is `[M, P]`.
  Undefined if `compute_uv` is `False`.
v: Left singular vectors. If `full_matrices` is `False` then shape is `[N, N]`.
  If `full_matrices` is `True` then shape is `[N, P]`.
  Undefined if `compute_uv` is false.
compute_uv: If true, left and right singular vectors will be
  computed and returned in `u` and `v`, respectively.
  If false, `u` and `v` are not set and should never referenced.
full_matrices: If true, compute full-sized `u` and `v`. If false
  (the default), compute only the leading `P` singular vectors.
  Ignored if `compute_uv` is `False`.
)doc");

REGISTER_OP("BatchSvd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float}")
    .SetShapeFn(BatchSvdShapeFn)
    .Doc(R"doc(
Computes the singular value decompositions of a batch of matrices.

Computes the SVD of each inner matrix in `input` such that
`input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`

```prettyprint
# a is a tensor containing a batch of matrices.
# s is a tensor of singular values for each matrix.
# u is the tensor containing of left singular vectors for each matrix.
# v is the tensor containing of right singular vectors for each matrix.
s, u, v = batch_svd(a)
s, _, _ = batch_svd(a, compute_uv=False)
```

input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
s: Singular values. Shape is `[..., P]`.
u: Left singular vectors. If `full_matrices` is `False` then shape is
  `[..., M, M]`; if `full_matrices` is `True` then shape is
  `[..., M, P]`. Undefined if `compute_uv` is `False`.
v: Left singular vectors. If `full_matrices` is `False` then shape is
  `[..., N, N]`. If `full_matrices` is `True` then shape is `[..., N, P]`.
  Undefined if `compute_uv` is false.
compute_uv: If true, left and right singular vectors will be
  computed and returned in `u` and `v`, respectively.
  If false, `u` and `v` are not set and should never referenced.
full_matrices: If true, compute full-sized `u` and `v`. If false
  (the default), compute only the leading `P` singular vectors.
  Ignored if `compute_uv` is `False`.
)doc");

}  // namespace tensorflow
