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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaSpmdFullToShardShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdFullToShardShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
  }

  ~XlaSpmdFullToShardShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    auto input_shape_or = ctx->InputXlaShape(0);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }
    auto output_shape = input_shape_or.ValueOrDie();
    int64 rank = output_shape.rank();
    if (sharding.type() == xla::OpSharding::OTHER) {
      for (int64 i = 0; i < rank; ++i) {
        int64 partitions_i = sharding.tile_assignment_dimensions(i);
        if (partitions_i == 1) continue;
        int64 dim_size =
            xla::CeilOfRatio(output_shape.dimensions(i), partitions_i);
        output_shape.set_dimensions(i, dim_size);
      }
    }
    xla::XlaOp input_annotation;
    {
      // Annotate the full-shape input with the sharding.
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(),
                                                       sharding);
      input_annotation =
          xla::CustomCall(ctx->builder(), /*call_target_name=*/"Sharding",
                          {input}, input_shape_or.ValueOrDie());
    }

    {
      // Annotate the shard-shape output with manual sharding, so that the
      // partitioner will leave it as is.
      xla::OpSharding manual;
      manual.set_type(xla::OpSharding::MANUAL);
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(), manual);
      auto output = xla::CustomCall(ctx->builder(),
                                    /*call_target_name=*/"SPMDFullToShardShape",
                                    {input_annotation}, output_shape);
      ctx->SetOutput(0, output);
    }
  }

 private:
  string manual_sharding_str_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdFullToShardShapeOp);
};

class XlaSpmdShardToFullShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdShardToFullShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_shape", &full_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
  }

  ~XlaSpmdShardToFullShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    auto input_shape_or = ctx->InputXlaShape(0);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto output_shape = TensorShapeToXLAShape(
        input_shape_or.ValueOrDie().element_type(), full_shape_);

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }
    xla::XlaOp input_annotation;
    {
      // Annotate the shard-shape input with manual sharding, so that the
      // partitioner will leave it as is.
      xla::OpSharding manual;
      manual.set_type(xla::OpSharding::MANUAL);
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(), manual);
      input_annotation =
          xla::CustomCall(ctx->builder(), /*call_target_name=*/"Sharding",
                          {input}, input_shape_or.ValueOrDie());
    }

    {
      // Annotate the full-shape output with the sharding.
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(),
                                                       sharding);
      ctx->SetOutput(
          0, xla::CustomCall(ctx->builder(),
                             /*call_target_name=*/"SPMDShardToFullShape",
                             {input_annotation}, output_shape));
    }
  }

 private:
  TensorShape full_shape_;
  string manual_sharding_str_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdShardToFullShapeOp);
};

REGISTER_XLA_OP(Name("XlaSpmdFullToShardShape"), XlaSpmdFullToShardShapeOp);
REGISTER_XLA_OP(Name("XlaSpmdShardToFullShape"), XlaSpmdShardToFullShapeOp);

}  // namespace
}  // namespace tensorflow
