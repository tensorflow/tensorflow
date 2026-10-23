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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/image_resizer_state.h"

namespace tensorflow {
namespace {

bool HasNonIntegerScale(int64_t in_size, int64_t out_size) {
  if (in_size <= 0 || out_size <= 0) return false;
  return (in_size % out_size != 0) && (out_size % in_size != 0);
}

struct ResizeDimensions {
  int64_t in_h = 0;
  int64_t in_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

struct ResizeNearestOptions {
  bool align_corners = false;
  bool half_pixel_centers = false;
};

// Returns, for each output index along one axis, the input index sampled by the
// CPU `ResizeNearestNeighbor` kernel
// (core/kernels/image/resize_nearest_neighbor_op.cc), so the gather lowering is
// bit-exact with it.
// Requires: 0 < in_size <= INT32_MAX + 1 and out_size >= 0.
std::vector<int32_t> ComputeNearestNeighborIndices(
    int64_t in_size, int64_t out_size, const ResizeNearestOptions& options) {
  DCHECK_GT(in_size, 0);
  DCHECK_GE(out_size, 0);
  DCHECK_LE(in_size, int64_t{std::numeric_limits<int32_t>::max()} + 1);
  std::vector<int32_t> indices(out_size);
  const float scale =
      CalculateResizeScale(in_size, out_size, options.align_corners);
  for (int64_t i = 0; i < out_size; ++i) {
    const float src = options.half_pixel_centers
                          ? (static_cast<float>(i) + 0.5f) * scale
                          : static_cast<float>(i) * scale;
    int64_t in_idx = std::clamp(
        static_cast<int64_t>(options.align_corners ? std::round(src)
                                                   : std::floor(src)),
        int64_t{0}, in_size - 1);
    DCHECK_LE(in_idx, std::numeric_limits<int32_t>::max());
    indices[i] = static_cast<int32_t>(in_idx);
  }
  return indices;
}

xla::XlaOp ResizeNearestWithGathers(XlaOpKernelContext& ctx, xla::XlaOp input,
                                    const ResizeDimensions& dims,
                                    const ResizeNearestOptions& options) {
  xla::XlaOp current = input;
  const auto resize_axis = [&](int64_t dim, int64_t in_size, int64_t out_size) {
    if (in_size == out_size) return;
    current = xla::TorchIndexSelect(
        current,
        xla::ConstantR1<int32_t>(
            ctx.builder(),
            ComputeNearestNeighborIndices(in_size, out_size, options)),
        dim);
  };

  // Shrink W before growing H so the HBM intermediate never exceeds
  // max(input, output); otherwise keep the measured H-then-W order.
  if (dims.out_h > dims.in_h && dims.out_w < dims.in_w) {
    resize_axis(/*dim=*/2, dims.in_w, dims.out_w);
    resize_axis(/*dim=*/1, dims.in_h, dims.out_h);
  } else {
    resize_axis(/*dim=*/1, dims.in_h, dims.out_h);
    resize_axis(/*dim=*/2, dims.in_w, dims.out_w);
  }
  return current;
}

}  // namespace

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

  std::string OpaqueField() const {
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

 protected:
  bool align_corners() const { return align_corners_; }
  bool half_pixel_centers() const { return half_pixel_centers_; }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

class TpuResizeNearestNeighborOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeNearestNeighborOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_VALUE(xla::Shape output_shape, ctx, GetOutputShape(ctx));
    OP_REQUIRES_VALUE(xla::Shape input_xla_shape, ctx, ctx->InputXlaShape(0));
    const TensorShape& input_tensor_shape = ctx->InputShape(0);

    const bool spatial_dims_static = input_tensor_shape.dims() == 4 &&
                                     !input_xla_shape.is_dynamic_dimension(1) &&
                                     !input_xla_shape.is_dynamic_dimension(2) &&
                                     !output_shape.is_dynamic_dimension(1) &&
                                     !output_shape.is_dynamic_dimension(2);

    // TF rejects align_corners && half_pixel_centers. The legacy custom call
    // rounds for that pair, so keep it there to leave its TPU output unchanged.
    if (!spatial_dims_static || (align_corners() && half_pixel_centers())) {
      CompileForward(ctx, "ResizeNearest");
      return;
    }

    const int64_t in_h = input_tensor_shape.dim_size(1);
    const int64_t in_w = input_tensor_shape.dim_size(2);
    const int64_t out_h = output_shape.dimensions(1);
    const int64_t out_w = output_shape.dimensions(2);

    const bool has_non_integer_scale =
        HasNonIntegerScale(in_h, out_h) || HasNonIntegerScale(in_w, out_w);
    constexpr int64_t kMaxIndexDimension =
        int64_t{std::numeric_limits<int32_t>::max()} + 1;

    // Identity, 1x1, empty, or over-INT32_MAX inputs fall back cleanly to
    // CompileForward paths.
    if (!has_non_integer_scale || in_h <= 0 || in_w <= 0 || out_h <= 0 ||
        out_w <= 0 || in_h > kMaxIndexDimension || in_w > kMaxIndexDimension) {
      CompileForward(ctx, "ResizeNearest");
      return;
    }

    const ResizeDimensions dims{
        .in_h = in_h,
        .in_w = in_w,
        .out_h = out_h,
        .out_w = out_w,
    };
    const ResizeNearestOptions options{
        .align_corners = align_corners(),
        .half_pixel_centers = half_pixel_centers(),
    };
    xla::XlaOp result =
        ResizeNearestWithGathers(*ctx, ctx->Input(0), dims, options);
    ctx->SetOutput(0, xla::ConvertElementType(result, ctx->output_xla_type(0)));
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
