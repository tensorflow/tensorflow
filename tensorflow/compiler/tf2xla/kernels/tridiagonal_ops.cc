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
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/lib/tridiagonal.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class TridiagonalSolveOp : public XlaOpKernel {
 public:
  explicit TridiagonalSolveOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partial_pivoting", &pivoting_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, !pivoting_,
        errors::Unimplemented(
            "Pivoting is not yet supported in XLA tridiagonal solver."));

    auto diagonals = ctx->Input(0);
    auto rhs = ctx->Input(1);

    auto result = xla::tridiagonal::ThomasSolver(diagonals, rhs);
    if (!result.ok()) {
      ctx->SetStatus(result.status());
      return;
    }
    ctx->SetOutput(0, result.ValueOrDie());
  }

 private:
  bool pivoting_;
};

// TODO(belletti): address test breakage in tridiagonal_solve_op_test_xla_gpu.py
// to support all XLA devices.
REGISTER_XLA_OP(Name("TridiagonalSolve")
                    .Device("XLA_TPU_JIT")
                    .TypeConstraint("T", kFloatTypes),
                TridiagonalSolveOp);

}  // namespace
}  // namespace tensorflow
