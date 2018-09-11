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

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

class BroadcastToOp : public XlaOpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    TensorShape output_shape;
    OP_REQUIRES_OK(context, context->ConstantInputAsShape(1, &output_shape));

    OP_REQUIRES(context, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Input rank (", input_shape.dims(),
                    ") must be less than or equal to the output rank (",
                    output_shape.dims(), ")"));

    auto input_dims = input_shape.dim_sizes();
    auto output_dims = output_shape.dim_sizes();

    // Broadcasting is done right-to-left on right-aligned dimensions; reverse
    // the two vectors so elements to be broadcast are aligned.
    absl::c_reverse(input_dims);
    absl::c_reverse(output_dims);

    std::vector<int64> broadcast_dims;
    std::vector<int64> broadcast_shape;
    for (int i = 0; i < output_shape.dims(); ++i) {
      if (i < input_shape.dims()) {
        OP_REQUIRES(
            context,
            (output_dims[i] == 0 && input_dims[i] == 0) ||
                (input_dims[i] != 0 && output_dims[i] % input_dims[i] == 0),
            errors::InvalidArgument("invalid shape to broadcast from ",
                                    input_shape.DebugString(), " to ",
                                    output_shape.DebugString()));

        broadcast_dims.push_back(broadcast_shape.size());
        if (output_dims[i] == input_dims[i] || input_dims[i] == 1) {
          broadcast_shape.push_back(output_dims[i]);
        }
        if (output_dims[i] != input_dims[i]) {
          // Add dimensions [I, O/I], which we will later flatten to just
          // [O]. We must do this in two phases since XLA broadcasting does not
          // support tiling.
          broadcast_shape.push_back(input_dims[i]);
          broadcast_shape.push_back(output_dims[i] / input_dims[i]);
        }
      } else {
        broadcast_shape.push_back(output_dims[i]);
      }
    }
    absl::c_reverse(broadcast_dims);
    int broadcast_shape_size = broadcast_shape.size();
    for (int64& broadcast_dim : broadcast_dims) {
      broadcast_dim = broadcast_shape_size - broadcast_dim - 1;
    }
    absl::c_reverse(broadcast_shape);
    xla::XlaOp output = xla::Reshape(
        xla::BroadcastInDim(context->Input(0),
                            xla::ShapeUtil::MakeShape(
                                context->input_xla_type(0), broadcast_shape),
                            broadcast_dims),
        output_shape.dim_sizes());
    context->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("BroadcastTo").CompileTimeConstInput("shape"),
                BroadcastToOp);

}  // namespace
}  // namespace tensorflow
