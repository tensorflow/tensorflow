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
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class AddNOp : public XlaOpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("AddN requires at least one argument"));

    XlaExpression::Kind kind = ctx->InputExpression(0).kind();
    xla::XlaOp sum;
    switch (kind) {
      case XlaExpression::Kind::kTensorList: {
        // Check that all TensorLists are initialized.
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          xla::XlaOp list = ctx->Input(i);
          bool is_initialized;
          OP_REQUIRES_OK(ctx, IsTensorListInitialized(list, &is_initialized));
          OP_REQUIRES(
              ctx, is_initialized,
              errors::InvalidArgument("TensorList input #", i,
                                      " for AddN op is an uninitialized list"));
        }
        // Nested TensorList is not supported.
        bool is_nested_list;
        OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested_list));
        OP_REQUIRES(ctx, !is_nested_list,
                    errors::Unimplemented(
                        "Nested TensorList is not supported for AddN op"));

        OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(0), &sum));
        xla::Shape sum_shape;
        OP_REQUIRES_OK(ctx,
                       GetTensorListBufferShape(ctx->Input(0), &sum_shape));
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          xla::XlaOp operand;
          OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(i), &operand));
          // Check that the shapes match.
          xla::Shape operand_shape;
          OP_REQUIRES_OK(
              ctx, GetTensorListBufferShape(ctx->Input(i), &operand_shape));
          OP_REQUIRES(
              ctx, sum_shape.dimensions() == operand_shape.dimensions(),
              errors::InvalidArgument(
                  "TensorList arguments to AddN must all have the same ",
                  "shape.\n", "Expected: ", sum_shape.DebugString(), "\n",
                  "Found: ", operand_shape.DebugString()));
          sum = xla::Add(sum, operand);
        }
        xla::XlaOp push_index;
        OP_REQUIRES_OK(ctx, GetTensorListPushIndex(ctx->Input(0), &push_index));
        OP_REQUIRES_OK(ctx, BuildNonNestedTensorList(sum, push_index, &sum));
        ctx->SetTensorListOutput(0, sum);
        break;
      }
      default:
        sum = ctx->Input(0);
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          sum = xla::Add(sum, ctx->Input(i));
        }
        ctx->SetOutput(0, sum);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AddNOp);
};

REGISTER_XLA_OP(Name("AddN").AllowVariantTypes(), AddNOp);

}  // namespace
}  // namespace tensorflow
