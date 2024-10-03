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
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/lib/tridiagonal.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class TridiagonalSolveOp : public XlaOpKernel {
 public:
  explicit TridiagonalSolveOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    auto diagonals = ctx->Input(0);
    auto rhs = ctx->Input(1);
    bool partial_pivoting = false;
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(def(), "partial_pivoting", &partial_pivoting));
    if (partial_pivoting) {
      ctx->SetStatus(errors::Unimplemented(
          "Current implementation does not yet support pivoting."));
      return;
    }

    auto result = xla::tridiagonal::TridiagonalSolver(xla::tridiagonal::kThomas,
                                                      diagonals, rhs);
    if (!result.ok()) {
      ctx->SetStatus(result.status());
      return;
    }
    ctx->SetOutput(0, result.value());
  }
};

class TridiagonalMatMulOp : public XlaOpKernel {
 public:
  explicit TridiagonalMatMulOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    auto upper_diagonal = ctx->Input(0);
    auto main_diagonal = ctx->Input(1);
    auto lower_diagonal = ctx->Input(2);
    auto rhs = ctx->Input(3);

    auto result = xla::tridiagonal::TridiagonalMatMul(
        upper_diagonal, main_diagonal, lower_diagonal, rhs);
    if (!result.ok()) {
      ctx->SetStatus(result.status());
      return;
    }
    ctx->SetOutput(0, result.value());
  }
};

REGISTER_XLA_OP(Name("TridiagonalSolve").TypeConstraint("T", kFloatTypes),
                TridiagonalSolveOp);

REGISTER_XLA_OP(Name("TridiagonalMatMul").TypeConstraint("T", kFloatTypes),
                TridiagonalMatMulOp);

}  // namespace
}  // namespace tensorflow
