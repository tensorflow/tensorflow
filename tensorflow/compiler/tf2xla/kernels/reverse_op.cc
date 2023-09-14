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

// XLA-specific reverse Op.

#include <vector>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class ReverseOp : public XlaOpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // r = tf.reverse(x, revdims)
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape revd_shape = ctx->InputShape(1);
    // Validate input sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(revd_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        revd_shape.DebugString()));
    OP_REQUIRES(ctx, revd_shape.num_elements() == x_shape.dims(),
                errors::InvalidArgument("axes ", revd_shape.DebugString(),
                                        " must have same number of elements as"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    if (revd_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    // XlaBuilder::Rev() requires concrete values for dimensions arg.
    xla::Literal lax;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &lax));

    std::vector<int64_t> dimensions;
    for (int d = 0; d < x_shape.dims(); ++d) {
      if (lax.Get<bool>({d})) {
        dimensions.push_back(d);
      }
    }

    ctx->SetOutput(0, xla::Rev(ctx->Input(0), dimensions));
  }
};

REGISTER_XLA_OP(Name("Reverse").CompileTimeConstantInput("dims"), ReverseOp);

class ReverseV2Op : public XlaOpKernel {
 public:
  explicit ReverseV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // r = tf.reverse(x, axes)
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape axes_shape = ctx->InputShape(1);
    // Validate input sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(axes_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        axes_shape.DebugString()));
    OP_REQUIRES(ctx, axes_shape.num_elements() <= x_shape.dims(),
                errors::InvalidArgument("axes ", axes_shape.DebugString(),
                                        " can not have more elements"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    // Reverse is a no-op if axes argument is empty.
    if (axes_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    // XlaBuilder::Rev() requires concrete values for dimensions arg.
    std::vector<int64_t> axes;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &axes));

    // witnessed_axes is used to ensure that the same axis is not marked to be
    // reversed multiple times.
    absl::InlinedVector<bool, 8> witnessed_axes(x_shape.dims(), false);

    for (int d = 0; d < axes.size(); ++d) {
      OP_REQUIRES(
          ctx, (-x_shape.dims() <= axes[d]) && (axes[d] < x_shape.dims()),
          errors::InvalidArgument(axes[d], " is out of range [-",
                                  x_shape.dims(), ", ", x_shape.dims(), ")."));
      // Axes can be negative and are shifted to the canonical index before
      // being lowered to HLO.
      if (axes[d] < 0) {
        axes[d] += x_shape.dims();
      }
      OP_REQUIRES(ctx, !witnessed_axes[axes[d]],
                  errors::InvalidArgument("canonicalized axis ", axes[d],
                                          " was repeated."));
      witnessed_axes[axes[d]] = true;
    }

    ctx->SetOutput(0, xla::Rev(ctx->Input(0), axes));
  }
};

REGISTER_XLA_OP(Name("ReverseV2").CompileTimeConstantInput("axis"),
                ReverseV2Op);

}  // namespace
}  // namespace tensorflow
