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
#include "tensorflow/compiler/tf2xla/lib/util.h"

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class SwitchOp : public XlaOpKernel {
 public:
  explicit SwitchOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->InputShape("pred")),
                errors::InvalidArgument(
                    "The second input must be a scalar, but it has shape ",
                    ctx->InputShape("pred").DebugString()));

    xla::Literal pred_literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput("pred", &pred_literal));
    bool pred = pred_literal.data<bool>()[0];
    ctx->SetOutput(pred ? 1 : 0, ctx->Input("data"));
  }
};

REGISTER_XLA_OP(
    Name("Switch").CompileTimeConstantInput("pred").CompilationOnly(),
    SwitchOp);

class MergeOp : public XlaOpKernel {
 public:
  explicit MergeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType index_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->expected_output_dtype(1),
                                                &index_type));

    for (int i = 0; i < ctx->num_inputs(); i++) {
      if (ctx->op_kernel_context()->has_input(i)) {
        ctx->SetOutput(0, ctx->Input(i));
        ctx->SetOutput(
            1, xla::ConstantR0WithType(ctx->Input(i).builder(), index_type, i));
        return;
      }
    }
  }
};

REGISTER_XLA_OP(Name("Merge").CompilationOnly(), MergeOp);

}  // anonymous namespace
}  // namespace tensorflow
