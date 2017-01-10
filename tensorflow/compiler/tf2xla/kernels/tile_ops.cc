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

// XLA-specific Tile Op.

#include <vector>
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class TileOp : public XlaOpKernel {
 public:
  explicit TileOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape multiples_shape = ctx->InputShape(1);

    OP_REQUIRES(
        ctx, IsLegacyVector(multiples_shape),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples_shape.DebugString()));
    OP_REQUIRES(ctx, input_shape.dims() == multiples_shape.num_elements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input_shape.dims(), " but got length ",
                    multiples_shape.dim_size(0)));
    const int input_dims = input_shape.dims();

    // If input is a scalar then multiples has 0 elements and this is
    // a NoOp.
    if (input_dims == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }

    xla::Literal literal;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &literal));

    // zero_element_result is true if the final shape has 0 elements,
    // i.e. if any of the input dimensions or multiples is zero.
    std::vector<int64> multiples_array(input_dims);
    std::vector<int64> output_shape;
    bool all_multiples_are_one = true;
    bool one_dimension_is_broadcasted_without_multiple = true;
    for (int i = 0; i < input_dims; ++i) {
      int multiple = xla::LiteralUtil::Get<int>(literal, {i});
      OP_REQUIRES(ctx, multiple,
                  errors::InvalidArgument("Expected multiples[", i,
                                          "] >= 0, but got ", multiple));
      int64 new_dim = input_shape.dim_size(i) * multiple;
      output_shape.push_back(new_dim);
      multiples_array[i] = multiple;
      all_multiples_are_one = all_multiples_are_one && multiple == 1;
      // If the multiple of a non-one dimensions is not one, then binary
      // operation broadcast semantics will not be sufficient to implement the
      // tile operation.
      one_dimension_is_broadcasted_without_multiple =
          one_dimension_is_broadcasted_without_multiple &&
          ((input_shape.dim_size(i) > 1 && multiple == 1) ||
           input_shape.dim_size(i) == 1);
    }
    auto input = ctx->Input(0);
    // If all multiples are 1, than the input is the same as the output.
    if (all_multiples_are_one) {
      ctx->SetOutput(0, input);
      return;
    }
    if (one_dimension_is_broadcasted_without_multiple) {
      // Create a constant Zero the size of the output shape to leverage binary
      // operation broadcast semantics.
      auto broadcasted_zero = ctx->builder()->Broadcast(
          XlaHelpers::Zero(ctx->builder(), ctx->input_type(0)), output_shape);
      ctx->SetOutput(0, ctx->builder()->Add(broadcasted_zero, input));
      return;
    }

    // First broadcast the requisite number of multiples along each
    // dimension. This prepends the broadcasted dimensions, so an
    // input of shape [2,3,1] broadcast with multiples [5,4,3] will
    // end up with shape [5,4,3,2,3,1].
    auto broadcasted = ctx->builder()->Broadcast(input, multiples_array);
    // Now flatten and reshape. The broadcasted dimensions are
    // paired with the original dimensions so in the above example
    // we flatten [0,3,1,4,2,5] then reshape to [10,12,3].
    std::vector<int64> flattened;
    for (int i = 0; i < output_shape.size(); ++i) {
      flattened.push_back(i);
      flattened.push_back(i + output_shape.size());
    }
    xla::ComputationDataHandle output =
        ctx->builder()->Reshape(broadcasted, flattened, output_shape);

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

REGISTER_XLA_OP("Tile", TileOp);

}  // namespace
}  // namespace tensorflow
