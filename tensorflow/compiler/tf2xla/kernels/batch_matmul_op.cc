/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"

namespace tensorflow {
namespace {

class BatchMatMulOp : public XlaOpKernel {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &adj_y_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto result = xla::BatchDot(MaybeConjugate(ctx->Input(0), adj_x_), adj_x_,
                                MaybeConjugate(ctx->Input(1), adj_y_), adj_y_);
    ctx->SetOutput(0, result);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

REGISTER_XLA_OP(Name("BatchMatMul"), BatchMatMulOp);
REGISTER_XLA_OP(Name("BatchMatMulV2"), BatchMatMulOp);

}  // namespace
}  // namespace tensorflow
