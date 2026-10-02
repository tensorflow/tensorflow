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

#include <cstdint>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class BroadcastToOp : public XlaOpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsShape(
                       1, &output_shape, xla::ValueInferenceMode::kUpperBound));
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        context, context->ResolveInputDynamismIntoPredVector(1, &dynamic_dims));

    // xla::BroadcastTo also tiles an input dimension into an output dimension
    // that is a multiple of it, but TensorFlow only broadcasts dimensions of
    // size 1. Check TensorFlow's rules, with the BroadcastTo kernel's errors,
    // for dimensions whose sizes are static; a dynamic size is only known by
    // its bound here.
    const TensorShape input_shape = context->InputShape(0);
    OP_REQUIRES(context, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Rank of input (", input_shape.dims(),
                    ") must be no greater than rank of output shape (",
                    output_shape.dims(), ")."));
    OP_REQUIRES_VALUE(xla::Shape input_xla_shape, context,
                      context->InputXlaShape(0));
    const int rank_difference = output_shape.dims() - input_shape.dims();
    for (int i = 0; i < input_shape.dims(); ++i) {
      const int64_t input_size = input_shape.dim_size(i);
      const int64_t output_size = output_shape.dim_size(i + rank_difference);
      OP_REQUIRES(context,
                  input_size == output_size || input_size == 1 ||
                      input_xla_shape.is_dynamic_dimension(i) ||
                      dynamic_dims[i + rank_difference],
                  errors::InvalidArgument(
                      "Incompatible shapes: ", input_shape.DebugString(),
                      " vs. ", output_shape.DebugString()));
    }

    auto output_status_or =
        BroadcastTo(context->Input(0), output_shape.dim_sizes());
    OP_REQUIRES_OK(context, output_status_or.status());
    auto output = output_status_or.value();
    for (int64_t dim = 0; dim < dynamic_dims.size(); ++dim) {
      if (dynamic_dims[dim]) {
        output = xla::SetDimensionSize(
            output,
            xla::Reshape(xla::Slice(context->Input(1), {dim}, {dim + 1}, {1}),
                         {}),
            dim);
      }
    }

    context->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("BroadcastTo").CompileTimeConstantInput("shape"),
                BroadcastToOp);

}  // namespace
}  // namespace tensorflow
