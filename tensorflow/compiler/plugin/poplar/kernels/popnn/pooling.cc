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

// XLA specific pooling ops.
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

using namespace xla::poplarplugin;

namespace tensorflow {
namespace {
xla::StatusOr<xla::Window> MakeWindow(const xla::Shape& input_shape,
                                      absl::Span<const int64> window_dimensions,
                                      absl::Span<const int64> window_strides,
                                      xla::Padding xla_padding,
                                      absl::Span<const int64> lhs_dilation,
                                      absl::Span<const int64> rhs_dilation) {
  TF_RETURN_IF_ERROR(
      xla::ValidatePaddingValues(xla::AsInt64Slice(input_shape.dimensions()),
                                 window_dimensions, window_strides));

  std::vector<std::pair<int64, int64>> padding =
      xla::MakePadding(xla::AsInt64Slice(input_shape.dimensions()),
                       window_dimensions, window_strides, xla_padding);

  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return Status::OK();
    } else {
      return xla::InvalidArgument(
          "%s", absl::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name,
                    "\nNumber of window dimensions: ", window_dimensions.size(),
                    "\nNumber of ", x_name, ": ", x, "\n"));
    }
  };
  TF_RETURN_IF_ERROR(verify_size(window_strides.size(), "window strides"));
  TF_RETURN_IF_ERROR(verify_size(padding.size(), "padding entries"));
  TF_RETURN_IF_ERROR(verify_size(lhs_dilation.size(), "lhs dilation factors"));
  TF_RETURN_IF_ERROR(verify_size(rhs_dilation.size(), "rhs dilation factors"));

  xla::Window window;
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return window;
}

// Superclass of pooling ops.
class PoolingOp : public XlaOpKernel, IpuOpKernel {
 public:
  PoolingOp(OpKernelConstruction* ctx, int num_spatial_dims,
            PoplibsOp::Op op_type)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        op_type_(op_type) {
    AddRequiredAttributesToMap();
    if (ctx->num_inputs() == 1) {
      std::vector<int32> ksize_int;
      std::vector<int32> stride_int;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_int));
      OP_REQUIRES(ctx, ksize_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_int));
      OP_REQUIRES(ctx, stride_int.size() == num_dims(),
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify ",
                                          num_dims(), " dimensions"));
      for (int i = 0; i < num_dims(); ++i) {
        ksize_.push_back(ksize_int[i]);
        stride_.push_back(stride_int[i]);
      }
    }
    Padding padding;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    padding_ = (padding == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto ksize_or_error = GetKernelSize(ctx);
    OP_REQUIRES_OK(ctx, ksize_or_error.status());
    std::vector<int64> ksize = ksize_or_error.ValueOrDie();

    auto stride_or_error = GetStride(ctx);
    OP_REQUIRES_OK(ctx, stride_or_error.status());
    std::vector<int64> stride = stride_or_error.ValueOrDie();

    const TensorShape tensor_in_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, tensor_in_shape.dims() == num_dims(),
                errors::InvalidArgument("Input to ", type_string(),
                                        " operator must have ", num_dims(),
                                        " dimensions"));
    xla::XlaOp input = ctx->Input(0);

    // Get the input shape
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::Shape input_shape = TensorShapeToXLAShape(input_type, tensor_in_shape);

    xla::XlaBuilder& b = *ctx->builder();

    auto window_or_error =
        MakeWindow(input_shape, ksize, stride, padding_, {}, {});
    OP_REQUIRES_OK(ctx, window_or_error.status());
    xla::Window window = window_or_error.ValueOrDie();
    attribute_map_.AddAttribute("window", window);

    // Infer the shape - pooling shape can be inferred using
    // InferReduceWindowShape where the computation is scalars.
    xla::Shape scalar_shape =
        xla::ShapeUtil::MakeShape(input_shape.element_type(), {});
    xla::ProgramShape scalar_comp_shape = xla::ShapeUtil::MakeProgramShape(
        {scalar_shape, scalar_shape}, scalar_shape);
    auto output_shape_or_error = xla::ShapeInference::InferReduceWindowShape(
        input_shape, scalar_shape, window, scalar_comp_shape);
    OP_REQUIRES_OK(ctx, output_shape_or_error.status());
    xla::Shape output_shape = output_shape_or_error.ValueOrDie();

    // Get the reduction dimensions.
    auto reduction_dims = GetPoolingReductionDims(window);

    xla::XlaOp output;
    if (reduction_dims.size()) {
      std::vector<xla::XlaOp> args = {input};
      output = xla::CustomCall(
          &b, GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, op_type_), args,
          output_shape, attribute_map_.Serialise());
    } else {
      // This is a no-op when there are no reducing dimensions.
      output = input;
    }

    ctx->SetOutput(0, output);
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

 protected:
  xla::StatusOr<std::vector<int64>> GetKernelSize(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return ksize_;
    }
    const TensorShape ksize_shape = ctx->InputShape(1);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(ksize_shape)) {
      return errors::InvalidArgument("ksize must be a vector, not shape ",
                                     ksize_shape.DebugString());
    }
    if (ksize_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window ksize field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64> ksize;
    auto status = ctx->ConstantInputAsIntVector(1, &ksize);
    if (!status.ok()) {
      return status;
    }
    return ksize;
  }

  xla::StatusOr<std::vector<int64>> GetStride(XlaOpKernelContext* ctx) {
    if (ctx->num_inputs() == 1) {
      return stride_;
    }
    const TensorShape stride_shape = ctx->InputShape(2);
    // Validate input sizes.
    if (!TensorShapeUtils::IsVector(stride_shape)) {
      return errors::InvalidArgument("stride must be a vector, not shape ",
                                     stride_shape.DebugString());
    }
    if (stride_shape.num_elements() != num_dims()) {
      return errors::InvalidArgument(
          "Sliding window stride field must "
          "specify ",
          num_dims(), " dimensions");
    }
    std::vector<int64> stride;
    auto status = ctx->ConstantInputAsIntVector(2, &stride);
    if (!status.ok()) {
      return status;
    }
    return stride;
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  xla::Padding padding_;
  PoplibsOp::Op op_type_;
};

class MaxPoolOp : public PoolingOp {
 public:
  MaxPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, num_spatial_dims, PoplibsOp::MaxPool) {}
};

class MaxPool2DOp : public MaxPoolOp {
 public:
  explicit MaxPool2DOp(OpKernelConstruction* ctx)
      : MaxPoolOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(Name("MaxPool").Device(DEVICE_IPU_XLA_JIT), MaxPool2DOp);
REGISTER_XLA_OP(Name("MaxPoolV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DOp);

// TODO - uncomment this for T5956/T6966.
// class MaxPool3DOp : public MaxPoolOp {
//  public:
//   explicit MaxPool3DOp(OpKernelConstruction* ctx)
//       : MaxPoolOp(ctx, /*num_spatial_dims=*/3) {}
// };
// REGISTER_XLA_OP(Name("MaxPool3D").Device(DEVICE_IPU_XLA_JIT), MaxPool3DOp);

class AvgPoolOp : public PoolingOp {
 public:
  AvgPoolOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : PoolingOp(ctx, num_spatial_dims, PoplibsOp::AvgPool) {}
};

class AvgPool2DOp : public AvgPoolOp {
 public:
  explicit AvgPool2DOp(OpKernelConstruction* ctx)
      : AvgPoolOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(Name("AvgPool").Device(DEVICE_IPU_XLA_JIT), AvgPool2DOp);

// TODO - uncomment this for T5956/T6966.
// class AvgPool3DOp : public AvgPoolOp {
//  public:
//   explicit AvgPool3DOp(OpKernelConstruction* ctx)
//       : AvgPoolOp(ctx, /*num_spatial_dims=*/3) {}
// };
// REGISTER_XLA_OP(Name("AvgPool3D").Device(DEVICE_IPU_XLA_JIT), AvgPool3DOp);

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class MaxPoolGradOp : public XlaOpKernel, IpuOpKernel {
 public:
  MaxPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    AddRequiredAttributesToMap();
    if (ctx->num_inputs() == 3) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    if (ctx->num_inputs() != 3) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 5,
          errors::InvalidArgument("Must supply ksize and stride arguments."));
      const TensorShape ksize_shape = ctx->InputShape(3);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(ksize_shape),
                  errors::InvalidArgument("ksize must be a vector, not shape",
                                          ksize_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(3, &ksize_));

      const TensorShape stride_shape = ctx->InputShape(4);
      // Validate input sizes.
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(stride_shape),
                  errors::InvalidArgument("stride must be a vector, not shape",
                                          stride_shape.DebugString()));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(4, &stride_));
    }

    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));

    const TensorShape tensor_in_shape = ctx->InputShape(0);
    const TensorShape tensor_out_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // For maxpooling, tensor_in should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_in_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_in must be ", num_dims(),
                                        "-dimensional"));
    OP_REQUIRES(ctx, tensor_out_shape.dims() == num_dims(),
                errors::InvalidArgument("tensor_out must be ", num_dims(),
                                        "-dimensional"));
    // For maxpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    auto input = ctx->Input(0);
    auto out = ctx->Input(1);
    auto out_backprop = ctx->Input(2);

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::Shape input_shape = TensorShapeToXLAShape(input_type, tensor_in_shape);
    xla::XlaBuilder& b = *ctx->builder();

    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    auto window_or_error =
        MakeWindow(input_shape, ksize_, stride_, xla_padding, {}, {});
    OP_REQUIRES_OK(ctx, window_or_error.status());
    xla::Window window = window_or_error.ValueOrDie();
    attribute_map_.AddAttribute("window", window);

    // Get the reduction dimensions.
    auto reduction_dims = GetPoolingReductionDims(window);

    xla::XlaOp input_backprop;
    if (reduction_dims.size()) {
      std::vector<xla::XlaOp> args = {input, out, out_backprop};
      input_backprop =
          xla::CustomCall(&b,
                          GetPoplibsCustomOpTargetString(
                              PoplibsOp::Popnn, PoplibsOp::MaxPoolGrad),
                          args, input_shape, attribute_map_.Serialise());
    } else {
      // The gradient is all 0s when we don't reduce.
      input_backprop = xla::Zeros(&b, input_shape);
    }
    ctx->SetOutput(0, input_backprop);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
};

class MaxPool2DGradOp : public MaxPoolGradOp {
 public:
  explicit MaxPool2DGradOp(OpKernelConstruction* ctx)
      : MaxPoolGradOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(Name("MaxPoolGrad").Device(DEVICE_IPU_XLA_JIT),
                MaxPool2DGradOp);
REGISTER_XLA_OP(Name("MaxPoolGradV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("ksize")
                    .CompileTimeConstantInput("strides"),
                MaxPool2DGradOp);

// TODO - uncomment this for T5956/T6966.
// class MaxPool3DGradOp : public MaxPoolGradOp {
//  public:
//   explicit MaxPool3DGradOp(OpKernelConstruction* ctx)
//       : MaxPoolGradOp(ctx, /*num_spatial_dims=*/3) {}
// };
// REGISTER_XLA_OP(Name("MaxPool3DGrad").Device(DEVICE_IPU_XLA_JIT),
// MaxPool3DGradOp);

// The operation to compute AvgPool gradients.
// It takes two inputs:
//   - The shape of the original input tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
class AvgPoolGradOp : public XlaOpKernel, IpuOpKernel {
 public:
  AvgPoolGradOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    AddRequiredAttributesToMap();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES(ctx, ksize_.size() == num_dims(),
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride_));
    OP_REQUIRES(ctx, stride_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape tensor_input_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &tensor_input_shape));
    const TensorShape tensor_out_backprop_shape = ctx->InputShape(1);

    // For avgpooling, tensor_in_shape should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_input_shape.dims() == num_dims(),
                errors::InvalidArgument("orig_tensor_input_shape must be ",
                                        num_dims(), "-dimensional"));

    // For avgpooling, out_backprop should have num_dims() dimensions.
    OP_REQUIRES(ctx, tensor_out_backprop_shape.dims() == num_dims(),
                errors::InvalidArgument("out_backprop must be ", num_dims(),
                                        "-dimensional"));

    auto out_backprop = ctx->Input(1);

    xla::PrimitiveType data_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(1), &data_type));
    xla::Shape input_shape =
        TensorShapeToXLAShape(data_type, tensor_input_shape);
    xla::XlaBuilder& b = *ctx->builder();

    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    auto window_or_error =
        MakeWindow(input_shape, ksize_, stride_, xla_padding, {}, {});
    OP_REQUIRES_OK(ctx, window_or_error.status());
    xla::Window window = window_or_error.ValueOrDie();
    attribute_map_.AddAttribute("window", window);

    // Get the reduction dimensions.
    auto reduction_dims = GetPoolingReductionDims(window);

    xla::XlaOp input_backprop;
    if (reduction_dims.size()) {
      std::vector<xla::XlaOp> args = {out_backprop};
      input_backprop =
          xla::CustomCall(&b,
                          GetPoplibsCustomOpTargetString(
                              PoplibsOp::Popnn, PoplibsOp::AvgPoolGrad),
                          args, input_shape, attribute_map_.Serialise());
    } else {
      // The gradient is all 0s when we don't reduce.
      input_backprop = xla::Zeros(&b, input_shape);
    }

    ctx->SetOutput(0, input_backprop);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

  const int num_spatial_dims_;
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
};

class AvgPool2DGradOp : public AvgPoolGradOp {
 public:
  explicit AvgPool2DGradOp(OpKernelConstruction* ctx)
      : AvgPoolGradOp(ctx, /*num_spatial_dims=*/2) {}
};
REGISTER_XLA_OP(Name("AvgPoolGrad")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("orig_input_shape"),
                AvgPool2DGradOp);

// TODO - uncomment this for T5956/T6966.
// class AvgPool3DGradOp : public AvgPoolGradOp {
//  public:
//   explicit AvgPool3DGradOp(OpKernelConstruction* ctx)
//       : AvgPoolGradOp(ctx, /*num_spatial_dims=*/3) {}
// };
// REGISTER_XLA_OP(
//     Name("AvgPool3DGrad").Device(DEVICE_IPU_XLA_JIT).CompileTimeConstantInput("orig_input_shape"),
//     AvgPool3DGradOp);

}  // anonymous namespace
}  // namespace tensorflow
