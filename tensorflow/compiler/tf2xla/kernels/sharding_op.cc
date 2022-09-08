/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/sharding_op_util.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class ShardingOp : public XlaOpKernel {
 public:
  explicit ShardingOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    std::vector<int32_t> unspecified_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unspecified_dims", &unspecified_dims));
    for (int32_t i32 : unspecified_dims) {
      unspecified_dims_.push_back(i32);
    }
  }

  ~ShardingOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input;
    {
      // The builder might create a broadcast from a constant, so we clear
      // sharding for the input.
      xla::XlaScopedShardingAssignment no_sharding(ctx->builder(),
                                                   std::nullopt);
      input = ctx->Input(0);
    }
    auto shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, shape_or.status());

    ctx->SetOutput(
        0, xla::CustomCall(
               ctx->builder(), /*call_target_name=*/"Sharding", {input},
               shape_or.value(),
               /*opaque=*/
               xla::sharding_op_util::EncodeAttributes(unspecified_dims_)));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ShardingOp);
  std::vector<int64_t> unspecified_dims_;
};

REGISTER_XLA_OP(Name("XlaSharding"), ShardingOp);

}  // namespace
}  // namespace tensorflow
