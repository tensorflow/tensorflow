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

#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"

namespace tensorflow {
namespace {


class XlaSpmdFullToShardShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdFullToShardShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &single_dim_));
    std::vector<int32_t> unspecified_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unspecified_dims", &unspecified_dims));
    for (int32_t i32 : unspecified_dims) {
      unspecified_dims_.push_back(i32);
    }
  }

  ~XlaSpmdFullToShardShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }

    auto status_or_output = ConvertSpmdFullToShardShape(
        ctx->builder(),
        /*input=*/input, /*single_dim=*/single_dim_,
        /*manual_sharding=*/sharding, /*unspecified_dims=*/unspecified_dims_);
    OP_REQUIRES_OK(ctx, status_or_output.status());
    ctx->SetOutput(0, status_or_output.ValueOrDie());
  }

 private:
  string manual_sharding_str_;
  int32 single_dim_;
  std::vector<int64_t> unspecified_dims_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdFullToShardShapeOp);
};

class XlaSpmdShardToFullShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdShardToFullShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_shape", &full_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &single_dim_));
    std::vector<int32_t> unspecified_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unspecified_dims", &unspecified_dims));
    for (int32_t i32 : unspecified_dims) {
      unspecified_dims_.push_back(i32);
    }
  }

  ~XlaSpmdShardToFullShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    auto status_or_input_shape = ctx->InputXlaShape(0);
    OP_REQUIRES_OK(ctx, status_or_input_shape.status());
    const xla::Shape output_shape = TensorShapeToXLAShape(
        /*type=*/status_or_input_shape.ValueOrDie().element_type(),
        /*tensor_shape=*/full_shape_);

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }

    auto status_or_output = ConvertSpmdShardToFullShape(
        ctx->builder(),
        /*input=*/input, /*output_shape=*/output_shape,
        /*single_dim=*/single_dim_,
        /*manual_sharding=*/sharding, /*unspecified_dims=*/unspecified_dims_);
    OP_REQUIRES_OK(ctx, status_or_output.status());
    ctx->SetOutput(0, status_or_output.ValueOrDie());
  }

 private:
  TensorShape full_shape_;
  string manual_sharding_str_;
  int32 single_dim_;
  std::vector<int64_t> unspecified_dims_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdShardToFullShapeOp);
};

REGISTER_XLA_OP(Name("XlaSpmdFullToShardShape"), XlaSpmdFullToShardShapeOp);
REGISTER_XLA_OP(Name("XlaSpmdShardToFullShape"), XlaSpmdShardToFullShapeOp);

}  // namespace
}  // namespace tensorflow
