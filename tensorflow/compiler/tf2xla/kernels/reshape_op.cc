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

// XLA-specific reshape Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class ReshapeOp : public XlaOpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape sizes_shape = ctx->InputShape(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, IsLegacyVector(sizes_shape),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes_shape.DebugString()));
    const int64 num_dims = sizes_shape.num_elements();

    xla::Literal literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &literal));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one if there
    // is one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    for (int d = 0; d < num_dims; ++d) {
      const int32 size = literal.Get<int>({d});
      if (size == -1) {
        OP_REQUIRES(
            ctx, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else {
        OP_REQUIRES(ctx, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          ctx, product > 0,
          errors::InvalidArgument("Reshape cannot infer the missing input size "
                                  "for an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int64 missing = input_shape.num_elements() / product;
      OP_REQUIRES(
          ctx, product * missing == input_shape.num_elements(),
          errors::InvalidArgument(
              "Input to reshape is a tensor with ", input_shape.num_elements(),
              " values, but the requested shape requires a multiple of ",
              product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(ctx, shape.num_elements() == input_shape.num_elements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input_shape.num_elements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    VLOG(1) << "Reshape " << input_shape.DebugString() << " "
            << shape.DebugString();

    ctx->SetOutput(0,
                   ctx->builder()->Reshape(ctx->Input(0), shape.dim_sizes()));
  }
};

REGISTER_XLA_OP(Name("Reshape").CompileTimeConstInput("shape"), ReshapeOp);

}  // namespace
}  // namespace tensorflow
