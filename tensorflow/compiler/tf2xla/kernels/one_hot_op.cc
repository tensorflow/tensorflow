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

// XLA implementation of OneHot operator.

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

class OneHotOp : public XlaOpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape indices_shape = ctx->InputShape(0);
    const TensorShape depth_shape = ctx->InputShape(1);
    const TensorShape on_value_shape = ctx->InputShape(2);
    const TensorShape off_value_shape = ctx->InputShape(3);

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth_shape),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value_shape),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value_shape),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value_shape.DebugString()));

    const int axis = (axis_ == -1) ? indices_dims : axis_;

    // The one-hot dimension.
    int64_t depth;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &depth));
    OP_REQUIRES(
        ctx, depth >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth));

    xla::XlaOp one_hot;
    OP_REQUIRES_OK(
        ctx, XlaHelpers::OneHot(ctx->builder(), depth, axis, input_type(0),
                                indices_shape, ctx->Input(0), ctx->Input(2),
                                ctx->Input(3), &one_hot));
    ctx->SetOutput(0, one_hot);
  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotOp);
};

REGISTER_XLA_OP(Name("OneHot").CompileTimeConstantInput("depth"), OneHotOp);

}  // namespace
}  // namespace tensorflow
