/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific Ops for 2D convolution.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

class ConvOp : public XlaOpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, strides_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    int batch_dim = GetTensorBatchDimIndex(num_dims(), data_format_);
    int feature_dim = GetTensorFeatureDimIndex(num_dims(), data_format_);
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));

    const TensorShape input_shape = ctx->InputShape(0);
    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, ..., in_depth, out_depth]
    const TensorShape filter_shape = ctx->InputShape(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(
        ctx, input_shape.dims() == num_dims(),
        errors::InvalidArgument("input must be ", num_dims(), "-dimensional",
                                input_shape.DebugString()));
    OP_REQUIRES(
        ctx, filter_shape.dims() == num_dims(),
        errors::InvalidArgument("filter must be ", num_dims(),
                                "-dimensional: ", filter_shape.DebugString()));

    // The last two dimension of the filter are the input and output shapes.
    const int64 in_depth = filter_shape.dim_size(num_spatial_dims_);

    // The 'C' dimension for input is in_depth. It must be the same as
    // the filter's in_depth.
    OP_REQUIRES(ctx, in_depth == input_shape.dim_size(feature_dim),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", input_shape.dim_size(feature_dim)));

    xla::ConvolutionDimensionNumbers dims;
    std::vector<int64> window_strides;

    dims.set_batch_dimension(GetTensorBatchDimIndex(num_dims(), data_format_));
    dims.set_feature_dimension(feature_dim);
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      dims.add_spatial_dimensions(input_dim);
      dims.add_kernel_spatial_dimensions(i);
      window_strides.push_back(strides_.at(input_dim));
    }
    dims.set_kernel_input_feature_dimension(num_spatial_dims_);
    dims.set_kernel_output_feature_dimension(num_spatial_dims_ + 1);

    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    xla::ComputationDataHandle conv = ctx->builder()->ConvWithGeneralDimensions(
        ctx->Input(0), ctx->Input(1), window_strides, xla_padding, dims);
    ctx->SetOutput(0, conv);
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvOp);
};

class Conv2DOp : public ConvOp {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP("Conv2D", Conv2DOp);

class Conv3DOp : public ConvOp {
 public:
  explicit Conv3DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP("Conv3D", Conv3DOp);

// Backprop for input.
class ConvBackpropInputOp : public XlaOpKernel {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, strides_.size() == num_dims(),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    int batch_dim = GetTensorBatchDimIndex(num_dims(), data_format_);
    int feature_dim = GetTensorFeatureDimIndex(num_dims(), data_format_);
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));

    TensorShape input_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &input_shape));

    const TensorShape filter_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // Reuse dimension computation logic from conv_grad_ops.cc.
    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, ConvBackpropComputeDimensions(
                 type_string(), num_spatial_dims_, input_shape, filter_shape,
                 out_backprop_shape, strides_, padding_, data_format_, &dims));

    auto filter = ctx->Input(1);
    auto out_backprop = ctx->Input(2);

    // The input gradients are computed by a convolution of the output
    // gradients and the filter, with some appropriate padding. See the
    // comment at the top of conv_grad_ops.h for details.

    xla::ConvolutionDimensionNumbers dnums;
    dnums.set_batch_dimension(batch_dim);
    dnums.set_feature_dimension(feature_dim);

    // TF filter shape is [ H, W, ..., inC, outC ]
    // Transpose the input and output features for computing the gradient.
    dnums.set_kernel_input_feature_dimension(num_spatial_dims_ + 1);
    dnums.set_kernel_output_feature_dimension(num_spatial_dims_);

    std::vector<int64> kernel_spatial_dims(num_spatial_dims_);
    std::vector<std::pair<int64, int64>> padding(num_spatial_dims_);
    std::vector<int64> lhs_dilation(num_spatial_dims_);
    std::vector<int64> ones(num_spatial_dims_, 1);
    for (int i = 0; i < num_spatial_dims_; ++i) {
      dnums.add_spatial_dimensions(
          GetTensorSpatialDimIndex(num_dims(), data_format_, i));
      dnums.add_kernel_spatial_dimensions(i);

      kernel_spatial_dims[i] = i;
      padding[i] = {dims.spatial_dims[i].pad_before,
                    dims.spatial_dims[i].pad_after};
      lhs_dilation[i] = dims.spatial_dims[i].stride;
    }

    // Mirror the filter in the spatial dimensions.
    xla::ComputationDataHandle mirrored_weights =
        ctx->builder()->Rev(filter, kernel_spatial_dims);

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    xla::ComputationDataHandle in_backprop = ctx->builder()->ConvGeneralDilated(
        out_backprop, mirrored_weights, /*window_strides=*/ones, padding,
        lhs_dilation, /*rhs_dilation=*/ones, dnums);

    ctx->SetOutput(0, in_backprop);
  }

 protected:
  const int num_spatial_dims_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropInputOp);
};

class Conv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP("Conv2DBackpropInput", Conv2DBackpropInputOp);

class Conv3DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP("Conv3DBackpropInputV2", Conv3DBackpropInputOp);

class ConvBackpropFilterOp : public XlaOpKernel {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* ctx, int num_spatial_dims)
      : XlaOpKernel(ctx), num_spatial_dims_(num_spatial_dims) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    const int n_dim = GetTensorBatchDimIndex(num_dims(), data_format_);
    const int c_dim = GetTensorFeatureDimIndex(num_dims(), data_format_);

    OP_REQUIRES(
        ctx, (strides_[n_dim] == 1 && strides_[c_dim] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    const TensorShape activations_shape = ctx->InputShape(0);
    TensorShape filter_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(1, &filter_shape));
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // Reuse dimension computation logic from conv_grad_ops.cc.
    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(ctx, ConvBackpropComputeDimensions(
                            type_string(), num_spatial_dims_, activations_shape,
                            filter_shape, out_backprop_shape, strides_,
                            padding_, data_format_, &dims));

    xla::ComputationDataHandle activations = ctx->Input(0);
    xla::ComputationDataHandle gradients = ctx->Input(2);

    // The filter gradients are computed by a convolution of the input
    // activations and the output gradients, with some appropriate padding.
    // See the comment at the top of conv_grad_ops.h for details.

    xla::ConvolutionDimensionNumbers dnums;

    // The activations (inputs) form the LHS of the convolution.
    // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
    // For the gradient computation, we flip the roles of the batch and
    // feature dimensions.
    // Each spatial entry has size in_depth * batch

    // Swap n_dim and c_dim in the activations.
    dnums.set_batch_dimension(c_dim);
    dnums.set_feature_dimension(n_dim);

    // The gradients become the RHS of the convolution.
    // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
    // where the batch becomes the input feature for the convolution.
    dnums.set_kernel_input_feature_dimension(n_dim);
    dnums.set_kernel_output_feature_dimension(c_dim);

    std::vector<std::pair<int64, int64>> padding(num_spatial_dims_);
    std::vector<int64> rhs_dilation(num_spatial_dims_);
    std::vector<int64> ones(num_spatial_dims_, 1);

    for (int i = 0; i < num_spatial_dims_; ++i) {
      int dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      dnums.add_spatial_dimensions(dim);
      dnums.add_kernel_spatial_dimensions(dim);

      // We will also need to pad the input with zeros such that after the
      // convolution, we get the right size for the filter.
      // The padded_in_rows should be such that when we convolve this with the
      // expanded_out_rows as a filter, we should get filter_rows back.
      //
      const int padded_in_size = dims.spatial_dims[i].expanded_output_size +
                                 dims.spatial_dims[i].filter_size - 1;

      // However it can be smaller than input_rows: in this
      // case it means some of the inputs are not used.
      //
      // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
      //
      // INPUT =  [ A  B  C ]
      //
      // FILTER = [ x y ]
      //
      // and the output will only have one column: a = A * x + B * y
      //
      // and input "C" is not used at all.
      //
      // We apply negative padding in this case.
      const int total_pad_in_size =
          padded_in_size - dims.spatial_dims[i].input_size;

      // + For the VALID padding, we don't pad anything on the top/left side
      //   and pad the bottom/right side with the remaining space.
      // + For the SAME padding, we pad top/left side the same as bottom/right
      //   side.
      //
      // In addition, if the padded input size is smaller than the input size,
      // we need to ignore some training elements of the input. We do this by
      // applying negative padding on the right/bottom.
      const int before_pad_in_size =
          (total_pad_in_size > 0 && padding_ == Padding::SAME)
              ? total_pad_in_size / 2
              : 0;

      padding[i] = {before_pad_in_size, total_pad_in_size - before_pad_in_size};
      rhs_dilation[i] = dims.spatial_dims[i].stride;
    }

    // Besides padding the input, we will also expand output_rows to
    //    expanded_out_rows = (output_rows - 1) * stride + 1
    // with zeros in between:
    //
    //      a . . . b . . . c . . . d . . . e
    //
    // This is done by specifying the window dilation factors in the
    // convolution HLO below.
    auto filter_backprop = ctx->builder()->ConvGeneralDilated(
        activations, gradients,
        /*window_strides=*/ones, padding, /*lhs_dilation=*/ones, rhs_dilation,
        dnums);

    // The layout of filter_backprop will match the layout of
    // padded_activations
    // and so will have layout: [out_feature, h, w, ..., in_feature]
    // Tensorflow filter shape is [ H, W, ..., inC, outC ], so we transpose the
    // output.
    std::vector<int64> transpose_dims;
    transpose_dims.reserve(num_dims());
    for (int i = 0; i < num_spatial_dims_; ++i) {
      transpose_dims.push_back(dnums.spatial_dimensions(i));
    }
    transpose_dims.push_back(c_dim);
    transpose_dims.push_back(n_dim);
    xla::ComputationDataHandle filter_backprop_reshaped =
        ctx->builder()->Transpose(filter_backprop, transpose_dims);
    ctx->SetOutput(0, filter_backprop_reshaped);
  }

 protected:
  int num_spatial_dims_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropFilterOp);
};

class Conv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }
};
REGISTER_XLA_OP("Conv2DBackpropFilter", Conv2DBackpropFilterOp);

class Conv3DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/3) {}
};
REGISTER_XLA_OP("Conv3DBackpropFilterV2", Conv3DBackpropFilterOp);

}  // namespace
}  // namespace tensorflow
