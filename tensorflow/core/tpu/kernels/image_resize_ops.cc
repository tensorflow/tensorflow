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

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

class TpuCustomResizeOp : public XlaOpKernel {
 public:
  explicit TpuCustomResizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  absl::StatusOr<xla::Shape> GetOutputShape(XlaOpKernelContext* ctx) const {
    std::vector<int64_t> out_size;
    auto status = ctx->ConstantInputAsIntVector(1, &out_size);
    CHECK_EQ(out_size.size(), 2) << status;
    TF_ASSIGN_OR_RETURN(xla::Shape input_shape, ctx->InputXlaShape(0));
    xla::Shape output_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(0));
    output_shape.set_dimensions(1, out_size[0]);
    output_shape.set_dimensions(2, out_size[1]);
    output_shape.set_dynamic_dimension(0, input_shape.is_dynamic_dimension(0));
    output_shape.set_dynamic_dimension(3, input_shape.is_dynamic_dimension(3));
    return output_shape;
  }

  string OpaqueField() const {
    return absl::StrCat("\"", align_corners_, half_pixel_centers_, "\"");
  }

  void CompileGrad(XlaOpKernelContext* ctx, const char* target,
                   const xla::Shape& output_shape) {
    auto input_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(0));
    if (ctx->InputShape(1).dim_sizes() == ctx->InputShape(0).dim_sizes()) {
      ctx->SetOutput(
          0, xla::ConvertElementType(ctx->Input(0), ctx->output_xla_type(0)));
      return;
    }
    // The gradient should be done in two phases for large resizes.
    auto input = ctx->Input(0);
    if (input_shape.dimensions(1) / output_shape.dimensions(1) > 3 &&
        input_shape.dimensions(2) / output_shape.dimensions(2) > 3) {
      auto intermediate_shape = output_shape;
      intermediate_shape.set_dimensions(1, input_shape.dimensions(1));
      input = xla::CustomCall(ctx->builder(), target, {ctx->Input(0)},
                              intermediate_shape, OpaqueField());
    }
    ctx->SetOutput(0, xla::CustomCall(ctx->builder(), target, {input},
                                      output_shape, OpaqueField()));
  }

  void CompileForward(XlaOpKernelContext* ctx, const char* target) {
    OP_REQUIRES_VALUE(auto output_shape, ctx, GetOutputShape(ctx));
    if (ctx->InputShape(0).dim_size(1) == output_shape.dimensions(1) &&
        ctx->InputShape(0).dim_size(2) == output_shape.dimensions(2)) {
      ctx->SetOutput(
          0, xla::ConvertElementType(ctx->Input(0), ctx->output_xla_type(0)));
      return;
    }
    if (ctx->InputShape(0).dim_size(1) == 1 &&
        ctx->InputShape(0).dim_size(2) == 1) {
      ctx->SetOutput(
          0, ctx->Input(0) +
                 xla::Zeros(ctx->builder(),
                            xla::ShapeUtil::MakeStaticShape(output_shape)));
      return;
    }
    ctx->SetOutput(0, xla::CustomCall(ctx->builder(), target, {ctx->Input(0)},
                                      output_shape, OpaqueField()));
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

class TpuResizeNearestNeighborOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeNearestNeighborOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    CompileForward(ctx, "ResizeNearest");
  }
};

class TpuResizeBilinearOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeBilinearOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    CompileForward(ctx, "ResizeBilinear");
  }
};

class TpuResizeNearestNeighborGradOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeNearestNeighborGradOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_VALUE(xla::Shape output_shape, ctx, GetOutputShape(ctx));
    CompileGrad(ctx, "ResizeNearestGrad", output_shape);
  }
};

class TpuResizeBilinearGradOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeBilinearGradOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_VALUE(xla::Shape input_shape, ctx, ctx->InputXlaShape(1));
    xla::Shape output_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(1));
    output_shape.set_dynamic_dimension(0, input_shape.is_dynamic_dimension(0));
    output_shape.set_dynamic_dimension(3, input_shape.is_dynamic_dimension(3));
    CompileGrad(ctx, "ResizeBilinearGrad", output_shape);
  }
};

REGISTER_XLA_OP(Name("ResizeNearestNeighbor")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeNearestNeighborOp);

REGISTER_XLA_OP(Name("ResizeNearestNeighborGrad")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeNearestNeighborGradOp);

REGISTER_XLA_OP(Name("ResizeBilinear")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeBilinearOp);

REGISTER_XLA_OP(Name("ResizeBilinearGrad").Device(DEVICE_TPU_XLA_JIT),
                TpuResizeBilinearGradOp);

}  // namespace tensorflow
