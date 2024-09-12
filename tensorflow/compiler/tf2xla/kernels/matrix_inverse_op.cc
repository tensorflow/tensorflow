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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/lib/qr.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class MatrixInverseOp : public XlaOpKernel {
 public:
  explicit MatrixInverseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint", &adjoint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    int64_t ndims = input_shape.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims));
    OP_REQUIRES(
        ctx, input_shape.dim_size(ndims - 2) == input_shape.dim_size(ndims - 1),
        errors::InvalidArgument("Input matrices must be squares, got",
                                input_shape.dim_size(ndims - 2),
                                " != ", input_shape.dim_size(ndims - 1)));

    xla::XlaOp input = xla::MaybeTransposeInMinorDims(ctx->Input(0), adjoint_);

    // TODO(b/111271662): Using LU decomposition instead of QR should be faster.
    xla::XlaOp q, r;
    QrExplicit(input, /*full_matrices=*/false, q, r);

    xla::XlaOp output =
        xla::TriangularSolve(r, xla::TransposeInMinorDims(q),
                             /*left_side=*/true,
                             /*lower=*/false, /*unit_diagonal=*/false,
                             /*transpose_a=*/
                             xla::TriangularSolveOptions::NO_TRANSPOSE);
    ctx->SetOutput(0, output);
  }

 private:
  bool adjoint_;

  MatrixInverseOp(const MatrixInverseOp&) = delete;
  void operator=(const MatrixInverseOp&) = delete;
};

// TODO(b/135640736): Allow this for integer and complex types.
REGISTER_XLA_OP(Name("MatrixInverse").TypeConstraint("T", kFloatTypes),
                MatrixInverseOp);

}  // namespace
}  // namespace tensorflow
