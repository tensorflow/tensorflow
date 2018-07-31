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

#include "tensorflow/compiler/tf2xla/lib/qr.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class QROp : public XlaOpKernel {
 public:
  explicit QROp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    bool full_matrices;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_matrices", &full_matrices));
    OP_REQUIRES(
        ctx, full_matrices,
        errors::Unimplemented("full_matrices=False case of QR decomposition is "
                              "not implemented in TF/XLA"));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto result = QRDecomposition(ctx->Input(0));
    if (!result.ok()) {
      ctx->SetStatus(result.status());
      return;
    }
    ctx->SetOutput(0, result.ValueOrDie().q);
    ctx->SetOutput(1, result.ValueOrDie().r);
  }
};

REGISTER_XLA_OP(Name("Qr").TypeConstraint("T", kFloatTypes), QROp);

}  // namespace
}  // namespace tensorflow
