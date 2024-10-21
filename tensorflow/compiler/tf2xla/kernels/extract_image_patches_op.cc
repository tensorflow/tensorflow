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

#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

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

    OP_REQUIRES(
        ctx, num_dims >= 3,
        errors::InvalidArgument("Kernel size must have at least 3 dimensions"));
    const int num_spatial_dims = num_dims - 2;

    OP_REQUIRES(ctx, strides_.size() == num_dims,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims, " dimensions"));
    OP_REQUIRES(ctx, dilations_.size() == num_dims,
                errors::InvalidArgument("Dilations field must "
                                        "specify ",
                                        num_dims, " dimensions"));

    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
    OP_REQUIRES(
        ctx, ksizes_[batch_dim] == 1 && ksizes_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "kernel sizes > 1 in the batch and depth "
                              "dimensions."));
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not support "
                              "dilations in the batch and depth dimensions."));

    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      OP_REQUIRES(
          ctx, ksizes_[input_dim] >= 0,
          errors::Unimplemented("Kernel size values must be non-negative; ", i,
                                "th spatial dimension had dilation ",
                                dilations_[input_dim]));
      OP_REQUIRES(ctx, strides_[input_dim] >= 1,
                  errors::Unimplemented("Stride values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
      OP_REQUIRES(ctx, dilations_[input_dim] >= 1,
                  errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
    }

    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &type));

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == num_dims,
        errors::InvalidArgument("input must be ", num_dims, "-dimensional",
                                input_shape.DebugString()));
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

    xla::XlaOp conv =
        xla::ConvGeneralDilated(ctx->Input(0), filter, window_strides, padding,
                                lhs_dilation, rhs_dilation, dims, depth);
    // Feature group convolution, will end up with the kernel_size change more
    // rapidly than the depth. Reshape, transpose and reshape to reorder them.
    std::vector<int64_t> conv_dims =
        xla::SpanToVector(builder->GetShape(conv).value().dimensions());
    conv_dims.back() = depth;
    conv_dims.push_back(kernel_size);
    conv = xla::TransposeInMinorDims(xla::Reshape(conv, conv_dims));
    conv_dims.pop_back();
    conv_dims.back() *= kernel_size;
    conv = xla::Reshape(conv, conv_dims);
    ctx->SetOutput(0, conv);
  }

 protected:
  std::vector<int32> ksizes_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
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
