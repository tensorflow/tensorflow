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
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class TileOp : public XlaOpKernel {
 public:
  explicit TileOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape multiples_shape = ctx->InputShape("multiples");

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(multiples_shape),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples_shape.DebugString()));
    OP_REQUIRES(ctx, input_shape.dims() == multiples_shape.num_elements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input_shape.dims(), " but got length ",
                    multiples_shape.dim_size(0)));
    const int input_dims = input_shape.dims();
    auto input = ctx->Input(0);
    // If input is a scalar then multiples has 0 elements and this is
    // a NoOp.
    if (input_dims == 0) {
      ctx->SetOutput(0, input);
      return;
    }

    std::vector<int64> multiples;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector("multiples", &multiples));
    std::vector<int64> output_dims(input_shape.dims());
    for (int64 i = 0; i < input_shape.dims(); ++i) {
      OP_REQUIRES(ctx, multiples[i] >= 0,
                  errors::InvalidArgument("Expected multiples[", i,
                                          "] >= 0, but got ", output_dims[i]));
      output_dims[i] = input_shape.dim_size(i) * multiples[i];
    }

    // If all multiples are 1, than the input is the same as the output.
    if (absl::c_all_of(multiples,
                       [](int64 multiple) { return multiple == 1; })) {
      ctx->SetOutput(0, input);
      return;
    }

    auto result = BroadcastTo(ctx->Input("input"), output_dims);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.ValueOrDie());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

REGISTER_XLA_OP(Name("Tile").CompileTimeConstantInput("multiples"), TileOp);

}  // namespace
}  // namespace tensorflow
