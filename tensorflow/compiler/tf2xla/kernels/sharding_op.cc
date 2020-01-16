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
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class ShardingOp : public XlaOpKernel {
 public:
  explicit ShardingOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  ~ShardingOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    auto shape =
        TensorShapeToXLAShape(ctx->input_xla_type(0), ctx->InputShape(0));
    ctx->SetOutput(
        0, xla::CustomCall(ctx->builder(), /*call_target_name=*/"Sharding",
                           {input}, shape));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ShardingOp);
};

REGISTER_XLA_OP(Name("XlaSharding"), ShardingOp);

}  // namespace
}  // namespace tensorflow
