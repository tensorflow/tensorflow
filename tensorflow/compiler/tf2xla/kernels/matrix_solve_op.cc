/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class MatrixSolveOp : public XlaOpKernel {
 public:
  explicit MatrixSolveOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint", &adjoint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape matrix_shape = ctx->InputShape(0);
    int64_t matrix_ndims = matrix_shape.dims();
    OP_REQUIRES(ctx, matrix_ndims >= 2,
                errors::InvalidArgument(
                    "Input matrix must have rank >= 2, got ", matrix_ndims));
    OP_REQUIRES(ctx,
                matrix_shape.dim_size(matrix_ndims - 2) ==
                    matrix_shape.dim_size(matrix_ndims - 1),
                errors::InvalidArgument(
                    "Input matrices must be square, got",
                    matrix_shape.dim_size(matrix_ndims - 2),
                    " != ", matrix_shape.dim_size(matrix_ndims - 1)));

    xla::XlaOp matrix = ctx->Input(0);
    xla::XlaOp rhs = ctx->Input(1);

    // TODO(b/111271662): Using LU decomposition instead of QR should be faster.
    xla::XlaOp q, r;
    xla::QrExplicit(matrix, /*full_matrices=*/false, q, r);

    xla::XlaOp inv =
        xla::TriangularSolve(r, xla::TransposeInMinorDims(q),
                             /*left_side=*/true,
                             /*lower=*/false, /*unit_diagonal=*/false,
                             /*transpose_a=*/
                             xla::TriangularSolveOptions::NO_TRANSPOSE);

    xla::XlaOp output =
        xla::BatchDot(inv, adjoint_, rhs,
                      /*transpose_y=*/false, xla::PrecisionConfig::HIGHEST);
    ctx->SetOutput(0, output);
  }

 private:
  bool adjoint_;

  MatrixSolveOp(const MatrixSolveOp&) = delete;
  void operator=(const MatrixSolveOp&) = delete;
};

// TODO(b/111271662): Support integer and complex types.
REGISTER_XLA_OP(Name("MatrixSolve").TypeConstraint("T", kFloatTypes),
                MatrixSolveOp);

}  // namespace
}  // namespace tensorflow
