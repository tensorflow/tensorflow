/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class MatrixTriangularSolveOp : public XlaOpKernel {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint", &adjoint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto result = TriangularSolve(
        ctx->builder(), ctx->Input(0), ctx->Input(1), /*left_side=*/true,
        /*lower=*/lower_, /*transpose_a=*/adjoint_, /*conjugate_a=*/adjoint_);
    if (!result.ok()) {
      ctx->SetStatus(result.status());
      return;
    }
    ctx->SetOutput(0, result.ValueOrDie());
  }

 private:
  bool lower_;
  bool adjoint_;
};

REGISTER_XLA_OP(Name("MatrixTriangularSolve"), MatrixTriangularSolveOp);

}  // namespace
}  // namespace tensorflow
