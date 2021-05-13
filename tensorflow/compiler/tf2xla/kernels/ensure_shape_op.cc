/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific ensure_shape Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class EnsureShapeOp : public XlaOpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape shape = ctx->InputShape(0);

    // valiate shape
    OP_REQUIRES(
        ctx, expected_shape_.IsCompatibleWith(shape),
        errors::InvalidArgument("Shape of tensor ", this->def().input(0), " ",
                                shape.DebugString(),
                                " is not compatible with expected shape ",
                                expected_shape_.DebugString(), "."));

    // If shape matches, outputs the tensor.
    ctx->SetOutput(0, ctx->Input(0));
  }

 private:
  PartialTensorShape expected_shape_;
};

REGISTER_XLA_OP(Name("EnsureShape"), EnsureShapeOp);

}  // namespace
}  // namespace tensorflow
