/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/logdet.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

absl::Status CheckSquareMatrix(const TensorShape& input_shape) {
  const int64_t ndims = input_shape.dims();
  if (ndims < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input must have rank >= 2, got ", ndims));
  }
  if (input_shape.dim_size(ndims - 2) != input_shape.dim_size(ndims - 1)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input matrices must be square, got ", input_shape.dim_size(ndims - 2),
        " != ", input_shape.dim_size(ndims - 1)));
  }
  return absl::OkStatus();
}

// Broadcasts a scalar to the batch shape of a [..., n, n] matrix input.
xla::XlaOp BroadcastScalarToBatch(xla::XlaOp scalar,
                                  const TensorShape& input_shape) {
  std::vector<int64_t> batch_dims(input_shape.dims() - 2);
  for (int i = 0; i < input_shape.dims() - 2; ++i) {
    batch_dims[i] = input_shape.dim_size(i);
  }
  return xla::Broadcast(scalar, batch_dims);
}

// slogdet(A) = (sign, log|det|). For n == 0, det is defined to be 1.
// xla::SLogDet cannot handle that case: it slices Householder taus to n-1.
xla::SignAndLogDet SLogDetOrEmpty(XlaOpKernelContext* ctx,
                                  const TensorShape& input_shape) {
  const int64_t n = input_shape.dim_size(input_shape.dims() - 1);
  if (n == 0) {
    const xla::PrimitiveType type = ctx->input_xla_type(0);
    return xla::SignAndLogDet{
        BroadcastScalarToBatch(xla::One(ctx->builder(), type), input_shape),
        BroadcastScalarToBatch(xla::Zero(ctx->builder(), type), input_shape)};
  }
  return xla::SLogDet(ctx->Input(0));
}

class MatrixDeterminantOp : public XlaOpKernel {
 public:
  explicit MatrixDeterminantOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES_OK(ctx, CheckSquareMatrix(input_shape));

    // det = sign * exp(log|det|). Do not use xla::LogDet(): that returns NaN
    // for matrices with a negative determinant.
    const xla::SignAndLogDet slogdet = SLogDetOrEmpty(ctx, input_shape);
    ctx->SetOutput(0, slogdet.sign * xla::Exp(slogdet.logdet));
  }

 private:
  MatrixDeterminantOp(const MatrixDeterminantOp&) = delete;
  void operator=(const MatrixDeterminantOp&) = delete;
};

class LogMatrixDeterminantOp : public XlaOpKernel {
 public:
  explicit LogMatrixDeterminantOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES_OK(ctx, CheckSquareMatrix(input_shape));

    const xla::SignAndLogDet slogdet = SLogDetOrEmpty(ctx, input_shape);
    ctx->SetOutput(0, slogdet.sign);
    ctx->SetOutput(1, slogdet.logdet);
  }

 private:
  LogMatrixDeterminantOp(const LogMatrixDeterminantOp&) = delete;
  void operator=(const LogMatrixDeterminantOp&) = delete;
};

// TODO(b/135640736): Allow complex types once XLA QR/SLogDet is validated for
// them, matching MatrixInverse.
REGISTER_XLA_OP(Name("MatrixDeterminant").TypeConstraint("T", kFloatTypes),
                MatrixDeterminantOp);
REGISTER_XLA_OP(Name("LogMatrixDeterminant").TypeConstraint("T", kFloatTypes),
                LogMatrixDeterminantOp);

}  // namespace
}  // namespace tensorflow
