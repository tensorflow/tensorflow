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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// Return in <out> the result of making the end of <s> a square matrix.
Status MakeBatchSquareMatrix(InferenceContext* c, ShapeHandle input,
                             ShapeHandle* out) {
  ShapeHandle s;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &s));

  DimensionHandle d;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, -2), c->Dim(s, -1), &d));

  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(s, 0, -2, &batch_shape));
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(d, d), out));
  return Status::OK();
}

Status BatchUnchangedSquareShapeFn(InferenceContext* c) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &out));
  c->set_output(0, out);
  return Status::OK();
}

// The first input is [...,M,N] and second input is either [...,M,K] or [...,M].
// Output is [...,N,K] or [...,N]. If <square>, then input is [...,M,M].
Status MatrixSolveShapeFn(InferenceContext* c, bool square) {
  ShapeHandle lhs;
  ShapeHandle rhs;
  if (square) {
    TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  }
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  // Make the common batch subshape.
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  // Make sure the batch dimensions match between lhs and rhs.
  TF_RETURN_IF_ERROR(
      c->Merge(lhs_batch_shape, rhs_batch_shape, &lhs_batch_shape));

  DimensionHandle m;
  // lhs and rhs have the same value for m to be compatible.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, -2), c->Dim(rhs, -2), &m));
  DimensionHandle n = c->Dim(lhs, -1);
  if (square) {
    TF_RETURN_IF_ERROR(c->Merge(m, n, &n));
  }

  ShapeHandle out;
  // Build final shape (batch_shape + n + k) in <out>.
  TF_RETURN_IF_ERROR(c->Concatenate(lhs_batch_shape, c->Vector(n), &out));
  TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(c->Dim(rhs, -1)), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Input is [...,N,N]. Outputs are:
//   [...,N];[0], if compute_v is false,
//   [...,N];[...,N,N], if compute_v is true.
Status SelfAdjointEigV2ShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));
  DimensionHandle n;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, -2), c->Dim(input, -1), &n));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(n), &e_shape));
  c->set_output(0, e_shape);
  bool compute_v;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_v", &compute_v));
  if (compute_v) {
    ShapeHandle v_shape;
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(n, n), &v_shape));
    c->set_output(1, v_shape);
  } else {
    c->set_output(1, c->Vector(0ll));
  }
  return Status::OK();
}

// Input is [...,M,N].
// First and second outputs are:
//   [...,M,M]; [...,M,N], if full_matrices is true,
//   [...,M,P]; [...,P,N], if full_matrices is false,
// where P = min(M,N).
Status QrShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
  DimensionHandle m = c->Dim(input, -2);
  DimensionHandle n = c->Dim(input, -1);
  DimensionHandle p;
  TF_RETURN_IF_ERROR(c->Min(m, n, &p));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle q_shape;
  ShapeHandle r_shape;
  bool full_matrices;
  TF_RETURN_IF_ERROR(c->GetAttr("full_matrices", &full_matrices));
  if (full_matrices) {
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, m), &q_shape));
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, n), &r_shape));
  } else {
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, p), &q_shape));
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(p, n), &r_shape));
  }
  c->set_output(0, q_shape);
  c->set_output(1, r_shape);
  return Status::OK();
}

// Input is [...,M,N].  First output is [...,min(M,N)].
// Second and third outputs are:
//   [0]; [0], if compute_uv is false.
//   [...,M,M]; [...,N,N], if compute_uv is true and full_matrices is true,
//   [...,M,P]; [...,N,P], if compute_uv is true and full_matrices is false,
// where P = min(M,N).
Status SvdShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
  DimensionHandle m = c->Dim(input, -2);
  DimensionHandle n = c->Dim(input, -1);
  DimensionHandle p;
  TF_RETURN_IF_ERROR(c->Min(m, n, &p));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(p), &e_shape));
  c->set_output(0, e_shape);
  bool compute_uv;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_uv", &compute_uv));
  if (compute_uv) {
    ShapeHandle u_shape;
    ShapeHandle v_shape;
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

}  // namespace

REGISTER_OP("MatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(input, -2), &unused));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("LogMatrixDeterminant")
    .Input("input: T")
    .Output("sign: T")
    .Output("log_abs_determinant: T")
    .Attr("T: {float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(input, -2), &unused));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &s));
      c->set_output(0, s);

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &out));
      c->set_output(1, out);
      return Status::OK();
    });

REGISTER_OP("MatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("MatrixExponential")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("MatrixLogarithm")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("Cholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("CholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("SelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(11, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));

      DimensionHandle d = c->Dim(input, -1);
      DimensionHandle d_plus_1;
      TF_RETURN_IF_ERROR(c->Add(d, 1, &d_plus_1));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &s));
      TF_RETURN_IF_ERROR(c->Concatenate(s, c->Matrix(d_plus_1, d), &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("SelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(SelfAdjointEigV2ShapeFn);

REGISTER_OP("MatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return MatrixSolveShapeFn(c, true /* square (*/);
    });

REGISTER_OP("MatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return MatrixSolveShapeFn(c, true /* square (*/);
    });

REGISTER_OP("MatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .Attr("fast: bool = True")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle l2_regularizer;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &l2_regularizer));
      return MatrixSolveShapeFn(c, false /* square */);
    });

REGISTER_OP("Qr")
    .Input("input: T")
    .Output("q: T")
    .Output("r: T")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(QrShapeFn);

REGISTER_OP("Svd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(SvdShapeFn);

// Deprecated op registrations:

// Can be deleted after 3feb2017.
REGISTER_OP("BatchSelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(11, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn(shape_inference::UnknownShape);

// Can all be deleted after 9mar2017.
REGISTER_OP("BatchMatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double, complex64, complex128}")
    .Deprecated(13, "Use MatrixDeterminant instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixInverse instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use Cholesky instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchCholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Deprecated(13, "Use CholeskyGrad instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchSelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixSolve instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixTriangularSolve instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Attr("fast: bool = True")
    .Deprecated(13, "Use MatrixSolveLs instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchSvd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .Deprecated(13, "Use Svd instead.")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
