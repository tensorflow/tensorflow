/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

bool IsFloatingPointType(xla::PrimitiveType type) {
  return xla::primitive_util::IsFloatingPointType(type);
}

// Reorders feature-group convolution output to extract_patches layout.
xla::XlaOp ReshapeConvToPatches(xla::XlaBuilder* builder, xla::XlaOp conv,
                                int64_t depth, int64_t kernel_size) {
  std::vector<int64_t> conv_dims =
      xla::SpanToVector(builder->GetShape(conv).value().dimensions());
  conv_dims.back() = depth;
  conv_dims.push_back(kernel_size);
  conv = xla::TransposeInMinorDims(xla::Reshape(conv, conv_dims));
  conv_dims.pop_back();
  conv_dims.back() *= kernel_size;
  return xla::Reshape(conv, conv_dims);
}

class ExtractImagePatchesOp : public XlaOpKernel {
 public:
  explicit ExtractImagePatchesOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksizes", &ksizes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rates", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorFormat data_format = FORMAT_NHWC;
    const int num_dims = ksizes_.size();

    OP_REQUIRES(ctx, num_dims >= 3,
                absl::InvalidArgumentError(
                    "Kernel size must have at least 3 dimensions"));
    const int num_spatial_dims = num_dims - 2;

    OP_REQUIRES(ctx, strides_.size() == num_dims,
                absl::InvalidArgumentError(
                    absl::StrCat("Sliding window strides field must "
                                 "specify ",
                                 num_dims, " dimensions")));
    OP_REQUIRES(
        ctx, dilations_.size() == num_dims,
        absl::InvalidArgumentError(absl::StrCat("Dilations field must "
                                                "specify ",
                                                num_dims, " dimensions")));

    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
    OP_REQUIRES(
        ctx, ksizes_[batch_dim] == 1 && ksizes_[feature_dim] == 1,
        absl::UnimplementedError("Current implementation does not yet support "
                                 "kernel sizes > 1 in the batch and depth "
                                 "dimensions."));
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        absl::UnimplementedError("Current implementation does not yet support "
                                 "strides in the batch and depth dimensions."));
    OP_REQUIRES(ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
                absl::UnimplementedError(
                    "Current implementation does not support "
                    "dilations in the batch and depth dimensions."));

    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      OP_REQUIRES(
          ctx, ksizes_[input_dim] >= 0,
          absl::UnimplementedError(absl::StrCat(
              "Kernel size values must be non-negative; ", i,
              "th spatial dimension had dilation ", dilations_[input_dim])));
      OP_REQUIRES(
          ctx, strides_[input_dim] >= 1,
          absl::UnimplementedError(absl::StrCat(
              "Stride values must be positive; ", i,
              "th spatial dimension had dilation ", dilations_[input_dim])));
      OP_REQUIRES(
          ctx, dilations_[input_dim] >= 1,
          absl::UnimplementedError(absl::StrCat(
              "Dilation values must be positive; ", i,
              "th spatial dimension had dilation ", dilations_[input_dim])));
    }

    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &type));

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, input_shape.dims() == num_dims,
                absl::InvalidArgumentError(
                    absl::StrCat("input must be ", num_dims, "-dimensional",
                                 input_shape.DebugString())));
    const int64_t depth = input_shape.dim_size(feature_dim);

    xla::XlaBuilder* builder = ctx->builder();

    // The following code is equivalent to:
    // eye = np.eye(kH * kW * D).reshape([kH, kW, D, kH * kW * kD])
    int64_t kernel_size = 1;
    std::vector<int64_t> kernel_shape(num_dims, 1);
    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      kernel_shape[i] = ksizes_[input_dim];
      kernel_size *= ksizes_[input_dim];
    }
    kernel_shape[num_spatial_dims] = 1;
    kernel_shape[num_spatial_dims + 1] = kernel_size * depth;
    xla::Shape iota_kernel_shape =
        xla::ShapeUtil::MakeShape(xla::S32, {kernel_size, depth, kernel_size});
    xla::XlaOp pred_intermediate = xla::Eq(xla::Iota(builder, iota_kernel_shape,
                                                     /* iota_dimension= */ 0),
                                           xla::Iota(builder, iota_kernel_shape,
                                                     /* iota_dimension= */ 2));
    // In some cases TPU implementations give different results than CPU and GPU
    // when doing the conversion directly from pred to the final type. Add an
    // extra conversion to S32 here solves this.
    xla::XlaOp int_intermediate =
        xla::ConvertElementType(pred_intermediate, xla::S32);
    xla::XlaOp filter = xla::Reshape(
        xla::ConvertElementType(int_intermediate, type), kernel_shape);

    xla::ConvolutionDimensionNumbers dims;
    std::vector<int64_t> window_strides(num_spatial_dims);
    std::vector<int64_t> lhs_dilation(num_spatial_dims, 1);
    std::vector<int64_t> rhs_dilation(num_spatial_dims);
    std::vector<std::pair<int64_t, int64_t>> padding(num_spatial_dims);

    dims.set_input_batch_dimension(batch_dim);
    dims.set_output_batch_dimension(batch_dim);
    dims.set_input_feature_dimension(feature_dim);
    dims.set_output_feature_dimension(feature_dim);
    dims.set_kernel_input_feature_dimension(num_spatial_dims);
    dims.set_kernel_output_feature_dimension(num_spatial_dims + 1);

    for (int i = 0; i < num_spatial_dims; ++i) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      dims.add_input_spatial_dimensions(dim);
      dims.add_kernel_spatial_dimensions(i);
      dims.add_output_spatial_dimensions(dim);
      window_strides[i] = strides_.at(dim);
      rhs_dilation[i] = dilations_.at(dim);

      int64_t unused_output_size;
      OP_REQUIRES_OK(
          ctx, GetWindowedOutputSizeVerbose(
                   input_shape.dim_size(dim), ksizes_[dim], rhs_dilation[i],
                   window_strides[i], padding_, &unused_output_size,
                   &padding[i].first, &padding[i].second));
    }

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    if (IsFloatingPointType(type)) {
      // One-hot conv computes sum(w_i * x_i). IEEE 0*NaN poisons the sum even
      // when w_i=0, unlike eager im2col which copies a single index per patch.
      xla::XlaOp is_nan = xla::IsNan(input);
      xla::XlaOp zero =
          xla::ConvertElementType(xla::ConstantR0<int32_t>(builder, 0), type);
      xla::XlaOp input_safe = xla::Select(is_nan, zero, input);

      xla::XlaOp conv_out = xla::ConvGeneralDilated(
          input_safe, filter, window_strides, padding, lhs_dilation,
          rhs_dilation, dims, depth);
      conv_out = ReshapeConvToPatches(builder, conv_out, depth, kernel_size);

      xla::XlaOp is_nan_as_input_type =
          xla::ConvertElementType(is_nan, type);
      xla::XlaOp nan_indicator = xla::ConvGeneralDilated(
          is_nan_as_input_type, filter, window_strides, padding, lhs_dilation,
          rhs_dilation, dims, depth);
      nan_indicator =
          ReshapeConvToPatches(builder, nan_indicator, depth, kernel_size);

      xla::XlaOp nan_value =
          FloatLiteral(builder, type, std::numeric_limits<double>::quiet_NaN());
      output = xla::Select(
          xla::Gt(nan_indicator, zero),
          xla::Broadcast(nan_value,
                         builder->GetShape(conv_out).value().dimensions()),
          conv_out);
    } else {
      xla::XlaOp conv = xla::ConvGeneralDilated(
          input, filter, window_strides, padding, lhs_dilation, rhs_dilation,
          dims, depth);
      output = ReshapeConvToPatches(builder, conv, depth, kernel_size);
    }

    ctx->SetOutput(0, output);
  }

 protected:
  std::vector<int32_t> ksizes_;
  std::vector<int32_t> dilations_;
  std::vector<int32_t> strides_;
  Padding padding_;

 private:
  ExtractImagePatchesOp(const ExtractImagePatchesOp&) = delete;
  void operator=(const ExtractImagePatchesOp&) = delete;
};

// We don't support integers for the convolution for GPU used in the
// implementation of this op, so we limit the supported types.
REGISTER_XLA_CONV_OP(Name("ExtractImagePatches"), ExtractImagePatchesOp);

}  // namespace
}  // namespace tensorflow
