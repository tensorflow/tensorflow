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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class IdentityOp : public XlaOpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, ctx->Input(0));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IdentityOp);
};

// XLA_* devices also register a "real" Identity operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_XLA_OP(Name("Identity").CompilationOnly(), IdentityOp);

REGISTER_XLA_OP(Name("PreventGradient"), IdentityOp);
REGISTER_XLA_OP(Name("StopGradient"), IdentityOp);

}  // namespace
}  // namespace tensorflow
