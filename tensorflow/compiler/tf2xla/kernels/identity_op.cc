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

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class IdentityOp : public XlaOpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      if (IsTensorListInput(ctx, i)) {
        ctx->SetTensorListOutput(i, ctx->Input(i));
      } else {
        DCHECK(ctx->input_type(i) != DT_VARIANT);
        // Forwards using the underlying op_kernel_context so both tensor and
        // resource values are forwarded correctly.
        ctx->op_kernel_context()->set_output(
            i, ctx->op_kernel_context()->input(i));
      }
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IdentityOp);
};

// XLA_* devices also register a "real" Identity operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_XLA_OP(
    Name("Identity").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    IdentityOp);
REGISTER_XLA_OP(Name("IdentityN")
                    .AllowResourceTypes()
                    .AllowVariantTypes()
                    .CompilationOnly(),
                IdentityOp);
REGISTER_XLA_OP(Name("PlaceholderWithDefault"), IdentityOp);
REGISTER_XLA_OP(Name("PreventGradient"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("StopGradient").AllowVariantTypes(), IdentityOp);
REGISTER_XLA_OP(Name("Snapshot"), IdentityOp);
REGISTER_XLA_OP(Name("_EagerConst"), IdentityOp);

}  // namespace
}  // namespace tensorflow
