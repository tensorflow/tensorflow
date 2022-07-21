/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific ensure_shape Op.

#include <vector>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

class EnsureShapeOp : public XlaOpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape shape = ctx->InputShape(0);

    // valiate shape
    OP_REQUIRES(
        ctx, expected_shape_.IsCompatibleWith(shape),
        errors::InvalidArgument("Shape of tensor ", this->def().input(0), " ",
                                shape.DebugString(),
                                " is not compatible with expected shape ",
                                expected_shape_.DebugString(), "."));

    // If the shape dimension in `expected_shape_` is already static, we would
    // remove the dynamic dimensions in XLA dynamic padder.
    xla::XlaOp tensor = ctx->Input(0);
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPredVector(0, &dynamic_dims));
    for (int i = 0; i < expected_shape_.dims(); ++i) {
      if (expected_shape_.dim_size(i) > 0 && dynamic_dims[i]) {
        VLOG(1) << "RemoveDynamicDimension: " << i;
        tensor = xla::RemoveDynamicDimension(tensor, i);
      }
    }

    // If shape matches, outputs the tensor.
    ctx->SetOutput(0, tensor);
  }

 private:
  PartialTensorShape expected_shape_;
};

REGISTER_XLA_OP(Name("EnsureShape"), EnsureShapeOp);

}  // namespace
}  // namespace tensorflow
