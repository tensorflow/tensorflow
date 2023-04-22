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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

class ClipByValueOp : public XlaOpKernel {
 public:
  explicit ClipByValueOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape shape = ctx->InputShape(0);
    const TensorShape min_shape = ctx->InputShape(1);
    const TensorShape max_shape = ctx->InputShape(2);

    auto input = ctx->Input(0);
    auto min = ctx->Input(1);
    auto max = ctx->Input(2);

    auto shape_error = [&]() -> tensorflow::Status {
      return errors::InvalidArgument(
          "clip_value_min and clip_value_max must be either of "
          "the same shape as input, or a scalar. ",
          "Input shape: ", shape.DebugString(),
          " clip_value_min shape: ", min_shape.DebugString(),
          " clip_value_max shape: ", max_shape.DebugString());
    };

    if (shape != min_shape) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(min_shape), shape_error());
      min = xla::Broadcast(min, shape.dim_sizes());
    }
    if (shape != max_shape) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(max_shape), shape_error());
      max = xla::Broadcast(max, shape.dim_sizes());
    }
    ctx->SetOutput(0, xla::Clamp(min, input, max));
  }
};

REGISTER_XLA_OP(Name("ClipByValue"), ClipByValueOp);

}  // namespace
}  // namespace tensorflow
