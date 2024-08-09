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
#include "xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class XlaSelfAdjointEigOp : public XlaOpKernel {
 public:
  explicit XlaSelfAdjointEigOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_iter", &max_iter_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto result =
        xla::SelfAdjointEig(ctx->Input(0), lower_, max_iter_, epsilon_);
    ctx->SetOutput(0, result.w);
    ctx->SetOutput(1, result.v);
  }

 private:
  bool lower_;
  int32 max_iter_;
  float epsilon_;
};

class SelfAdjointEigV2Op : public XlaOpKernel {
 public:
  explicit SelfAdjointEigV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    int n = input_shape.dim_size(input_shape.dims() - 1);
    // This is based on heuristics that approx log(n) sweep updates are needed.
    // Note: the heuristics provides no theoretical guarantee, max_iter=100 and
    // epsilon should be used to determine exit condition.
    int max_iter = 2 * tensorflow::Log2Ceiling(n);
    auto result = xla::SelfAdjointEig(ctx->Input(0), true, max_iter, 1e-6);
    ctx->SetOutput(0, result.w);
    ctx->SetOutput(1, result.v);
  }
};

REGISTER_XLA_OP(Name("XlaSelfAdjointEig"), XlaSelfAdjointEigOp);
REGISTER_XLA_OP(Name("SelfAdjointEigV2"), SelfAdjointEigV2Op);

}  // namespace
}  // namespace tensorflow
