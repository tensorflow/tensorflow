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

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"

namespace tensorflow {
namespace {

class NextAfterOp : public XlaOpKernel {
 public:
  explicit NextAfterOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    auto lhs = ctx->Input(0);
    auto rhs = ctx->Input(1);
    OP_REQUIRES_OK(ctx, BroadcastOpsToSame(&lhs, &rhs));
    ctx->SetOutput(0, xla::NextAfter(lhs, rhs));
  }
};
REGISTER_XLA_OP(Name("NextAfter"), NextAfterOp);

}  // namespace
}  // namespace tensorflow
