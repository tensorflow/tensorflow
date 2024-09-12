/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>

#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

class ToBoolOp : public XlaOpKernel {
 public:
  explicit ToBoolOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, DoCompile(ctx));
  }

 private:
  Status DoCompile(XlaOpKernelContext* ctx) {
    auto input = ctx->Input(0);

    // If the input is a scalar, then non-zero value returns True.
    TF_ASSIGN_OR_RETURN(auto shape, ctx->InputXlaShape(0));
    if (shape.rank() == 0) {
      auto result = xla::Ne(ctx->Input(0), xla::ZerosLike(input));
      ctx->SetOutput(0, result);
      return absl::OkStatus();
    }

    // Otherwise, any input tensor with elements returns True. Input tensor
    // dimensions might be dynamic with bounds so multiply all the dimensions.
    xla::XlaOp num_elements = xla::One(ctx->builder(), xla::S32);
    for (int64_t dim = 0; dim < shape.rank(); dim++) {
      num_elements = xla::Mul(num_elements, xla::GetDimensionSize(input, dim));
    }
    auto result = xla::Ne(num_elements, xla::ZerosLike(num_elements));
    ctx->SetOutput(0, result);

    return absl::OkStatus();
  }
};

REGISTER_XLA_OP(Name("ToBool"), ToBoolOp);

}  // namespace
}  // namespace tensorflow
