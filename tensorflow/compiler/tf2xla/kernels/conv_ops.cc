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
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

// Returns the expanded size of a filter used for depthwise convolution.
// If `shape` is [H, W, ..., M, N] returns [H, W, ..., M, M*N].
TensorShape ExpandedFilterShapeForDepthwiseConvolution(
    const TensorShape& shape) {
  int num_dims = shape.dims();
  CHECK_GE(num_dims, 2);
  TensorShape expanded_shape = shape;
  expanded_shape.set_dim(num_dims - 1, shape.dim_size(num_dims - 2) *
                                           shape.dim_size(num_dims - 1));
  return expanded_shape;
}

// Broadcast zeros to ExpandedFilterShapeForDepthwiseConvolution.
xla::XlaOp CreateExpandedZero(const TensorShape& filter_shape, DataType dtype,
                              xla::XlaBuilder* builder) {
  TensorShape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
  return xla::Broadcast(XlaHelpers::Zero(builder, dtype),
                        expanded_filter_shape.dim_sizes());
}

// Create a mask for depthwise convolution that will make a normal convolution
// produce the same results as a depthwise convolution. For a [2, 2, 3, 2]
// depthwise filter this returns a [2, 2, 3, 6] tensor
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
// The first step is to create a one tensor, A, that is [3]
//   0 1 2
//
// and another tensor, B,  that is [3 * 2]
//   0 1 2 3 4 5
//
// and divide B it by 2 to get
//   0 0 1 1 2 2
//
// then we broadcast the B to [2, 2, 3, 3 * 2]
//   0 0 1 1 2 2   0 0 1 1 2 2
//   0 0 1 1 2 2   0 0 1 1 2 2
//   0 0 1 1 2 2   0 0 1 1 2 2
//
//   0 0 1 1 2 2   0 0 1 1 2 2
//   0 0 1 1 2 2   0 0 1 1 2 2
//   0 0 1 1 2 2   0 0 1 1 2 2
//
// Finally compare A and broadcasted B in dimension 2 amd return the result at
// the beginning of the comment.
xla::XlaOp CreateExpandedFilterMask(const TensorShape& filter_shape,
                                    xla::XlaBuilder* builder) {
  TensorShape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
  int64 depthwise_multiplier = filter_shape.dim_size(filter_shape.dims() - 1);
  int64 input_feature = filter_shape.dim_size(filter_shape.dims() - 2);

  // Create a M sized linspace and an M*N sized linspace that will be
  // broadcasted into perpendicular dimensions and compared.
  xla::XlaOp input_feature_iota = xla::Iota(builder, xla::S32, input_feature);
  xla::XlaOp expanded_feature_iota =
      xla::Iota(builder, xla::S32, input_feature * depthwise_multiplier);

  // Divide the M*N sized linspace by the depthwise_multiplier to create
  // [0 0 1 1 2 2] in the example in the function comment.
  expanded_feature_iota =
      xla::Div(expanded_feature_iota,
               XlaHelpers::IntegerLiteral(builder, DataType::DT_INT32,
                                          depthwise_multiplier));

  // Broadcast the N*M linspace to [H, W, ..., M, M*N].
  auto expanded_feature_broadcast_dims = expanded_filter_shape.dim_sizes();
  expanded_feature_broadcast_dims.pop_back();
  auto broadcasted_expanded_feature_iota =
      xla::Broadcast(expanded_feature_iota, expanded_feature_broadcast_dims);

  // Compare the broadcasted linspace to the input feature linspace in the
  // input feature dimension to create a diagonal predicate.
  return xla::Eq(broadcasted_expanded_feature_iota, input_feature_iota,
                 {expanded_filter_shape.dims() - 2});
}

// Expands a filter of shape [H, W, ..., M, N] to [H, W, ..., M, M*N] by adding
// zeros for the cross-depth filters. Used to build a depthwise convolution.
xla::XlaOp ExpandFilterForDepthwiseConvolution(const TensorShape& filter_shape,
                                               DataType dtype,
                                               const xla::XlaOp& filter,
                                               xla::XlaBuilder* builder) {
  int64 depthwise_multiplier = filter_shape.dim_size(filter_shape.dims() - 1);
  int64 input_feature = filter_shape.dim_size(filter_shape.dims() - 2);
  TensorShape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);

  // Create a [H, W, ..., 1, N*M] reshape of the filter.
  TensorShape implicit_broadcast_filter_shape = expanded_filter_shape;
  implicit_broadcast_filter_shape.set_dim(
      implicit_broadcast_filter_shape.dims() - 2, 1);
  implicit_broadcast_filter_shape.set_dim(
      implicit_broadcast_filter_shape.dims() - 1,
      depthwise_multiplier * input_feature);
  auto implicit_broadcast_filter =
      xla::Reshape(filter, implicit_broadcast_filter_shape.dim_sizes());

  // Broadcast the filter to  [H, W, ..., M, M*N].
  auto expanded_zero = CreateExpandedZero(filter_shape, dtype, builder);
  auto expanded_filter = xla::Add(implicit_broadcast_filter, expanded_zero);

  // If the filter mask is set, choose the broadcasted filter, othwerwise,
  // choose zero.
  return xla::Select(CreateExpandedFilterMask(filter_shape, builder),
                     expanded_filter, expanded_zero);
}

// Inverse of ExpandFilterForDepthwiseConvolution.
xla::XlaOp ContractFilterForDepthwiseBackprop(XlaOpKernelContext* ctx,
                                              const TensorShape& filter_shape,
                                              DataType dtype,
                                              const xla::XlaOp& filter_backprop,
                                              xla::XlaBuilder* builder) {
  TensorShape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
  auto masked_expanded_filter = xla::Select(
      CreateExpandedFilterMask(filter_shape, builder), filter_backprop,
      CreateExpandedZero(filter_shape, dtype, builder));
  return xla::Reshape(
      // This reduce does not need inputs to be converted with
      // XlaHelpers::SumAccumulationType() since the ExpandedFilterMask with
      // ExpandedZero guarantees that only one element is non zero, so there
      // cannot be accumulated precision error.
      xla::Reduce(masked_expanded_filter, XlaHelpers::Zero(builder, dtype),
                  *ctx->GetOrCreateAdd(dtype),
                  {expanded_filter_shape.dims() - 2}),
      filter_shape.dim_sizes());
}

class ConvOp : public XlaOpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* ctx, int num_spatial_dims,
                  bool depthwise)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        depthwise_(depthwise) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));

    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
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

    OP_REQUIRES(ctx, dilations_.size() == num_dims(),
                errors::InvalidArgument("Dilations field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(
        ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not support "
                              "dilations in the batch and depth dimensions."));
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      OP_REQUIRES(ctx, dilations_[input_dim] >= 1,
                  errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
    }

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

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp filter = ctx->Input(1);
    TensorShape expanded_filter_shape = filter_shape;
    if (depthwise_) {
      filter = ExpandFilterForDepthwiseConvolution(
          filter_shape, ctx->input_type(0), filter, b);
      expanded_filter_shape =
          ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
    }

    xla::ConvolutionDimensionNumbers dims;
    std::vector<int64> window_strides(num_spatial_dims_);
    std::vector<int64> lhs_dilation(num_spatial_dims_, 1);
    std::vector<int64> rhs_dilation(num_spatial_dims_);
    std::vector<std::pair<int64, int64>> padding(num_spatial_dims_);

    dims.set_input_batch_dimension(batch_dim);
    dims.set_output_batch_dimension(batch_dim);
    dims.set_input_feature_dimension(feature_dim);
    dims.set_output_feature_dimension(feature_dim);
    dims.set_kernel_input_feature_dimension(num_spatial_dims_);
    dims.set_kernel_output_feature_dimension(num_spatial_dims_ + 1);

    for (int i = 0; i < num_spatial_dims_; ++i) {
      const int64 dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      dims.add_input_spatial_dimensions(dim);
      dims.add_kernel_spatial_dimensions(i);
      dims.add_output_spatial_dimensions(dim);
      window_strides[i] = strides_.at(dim);
      rhs_dilation[i] = dilations_.at(dim);

      int64 unused_output_size;
      OP_REQUIRES_OK(
          ctx, GetWindowedOutputSizeVerboseV2(
                   input_shape.dim_size(dim), expanded_filter_shape.dim_size(i),
                   rhs_dilation[i], window_strides[i], padding_,
                   &unused_output_size, &padding[i].first, &padding[i].second));
    }

    xla::XlaOp conv =
        xla::ConvGeneralDilated(ctx->Input(0), filter, window_strides, padding,
                                lhs_dilation, rhs_dilation, dims);
    ctx->SetOutput(0, conv);
  }

 protected:
  const int num_spatial_dims_;
  const bool depthwise_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvOp);
};

class Conv2DOp : public ConvOp {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(Name("Conv2D"), Conv2DOp);

class Conv3DOp : public ConvOp {
 public:
  explicit Conv3DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(Name("Conv3D"), Conv3DOp);

class DepthwiseConv2DOp : public ConvOp {
 public:
  explicit DepthwiseConv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNative"), DepthwiseConv2DOp);

// Backprop for input.
class ConvBackpropInputOp : public XlaOpKernel {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* ctx, int num_spatial_dims,
                               bool depthwise)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        depthwise_(depthwise) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
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

    OP_REQUIRES(ctx, dilations_.size() == num_dims(),
                errors::InvalidArgument("Dilations field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(
        ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
        errors::Unimplemented("Current implementation does not support "
                              "dilations in the batch and depth dimensions."));
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      OP_REQUIRES(ctx, dilations_[input_dim] >= 1,
                  errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
    }

    TensorShape input_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &input_shape));

    const TensorShape filter_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    const TensorShape expanded_filter_shape =
        depthwise_ ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                   : filter_shape;
    // Reuse dimension computation logic from conv_grad_ops.cc.
    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(ctx,
                   ConvBackpropComputeDimensionsV2(
                       type_string(), num_spatial_dims_, input_shape,
                       expanded_filter_shape, out_backprop_shape, dilations_,
                       strides_, padding_, data_format_, &dims));

    xla::XlaBuilder* b = ctx->builder();
    auto filter = ctx->Input(1);
    auto out_backprop = ctx->Input(2);

    // The input gradients are computed by a convolution of the output
    // gradients and the filter, with some appropriate padding. See the
    // comment at the top of conv_grad_ops.h for details.

    xla::ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(batch_dim);
    dnums.set_output_batch_dimension(batch_dim);
    dnums.set_input_feature_dimension(feature_dim);
    dnums.set_output_feature_dimension(feature_dim);

    // TF filter shape is [ H, W, ..., inC, outC ]
    // Transpose the input and output features for computing the gradient.
    dnums.set_kernel_input_feature_dimension(num_spatial_dims_ + 1);
    dnums.set_kernel_output_feature_dimension(num_spatial_dims_);

    std::vector<int64> kernel_spatial_dims(num_spatial_dims_);
    std::vector<std::pair<int64, int64>> padding(num_spatial_dims_);
    std::vector<int64> lhs_dilation(num_spatial_dims_);
    std::vector<int64> rhs_dilation(num_spatial_dims_);
    std::vector<int64> ones(num_spatial_dims_, 1);
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int64 dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      dnums.add_input_spatial_dimensions(dim);
      dnums.add_kernel_spatial_dimensions(i);
      dnums.add_output_spatial_dimensions(dim);

      kernel_spatial_dims[i] = i;
      padding[i] = {dims.spatial_dims[i].pad_before,
                    dims.spatial_dims[i].pad_after};
      lhs_dilation[i] = dims.spatial_dims[i].stride;
      rhs_dilation[i] = dilations_[dim];
    }

    // If this is a depthwise convolution, expand the filter.
    if (depthwise_) {
      filter = ExpandFilterForDepthwiseConvolution(
          filter_shape, ctx->input_type(1), filter, b);
    }

    // Mirror the filter in the spatial dimensions.
    xla::XlaOp mirrored_weights = xla::Rev(filter, kernel_spatial_dims);

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    xla::XlaOp in_backprop = xla::ConvGeneralDilated(
        out_backprop, mirrored_weights, /*window_strides=*/ones, padding,
        lhs_dilation, rhs_dilation, dnums);

    ctx->SetOutput(0, in_backprop);
  }

 protected:
  const int num_spatial_dims_;
  const bool depthwise_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropInputOp);
};

class Conv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(
    Name("Conv2DBackpropInput").CompileTimeConstInput("input_sizes"),
    Conv2DBackpropInputOp);

class Conv3DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {}
};
REGISTER_XLA_OP(
    Name("Conv3DBackpropInputV2").CompileTimeConstInput("input_sizes"),
    Conv3DBackpropInputOp);

class DepthwiseConv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit DepthwiseConv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropInput")
                    .CompileTimeConstInput("input_sizes"),
                DepthwiseConv2DBackpropInputOp);

class ConvBackpropFilterOp : public XlaOpKernel {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* ctx, int num_spatial_dims,
                                bool depthwise)
      : XlaOpKernel(ctx),
        num_spatial_dims_(num_spatial_dims),
        depthwise_(depthwise) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  int num_dims() const { return num_spatial_dims_ + 2; }

  void Compile(XlaOpKernelContext* ctx) override {
    const int n_dim = GetTensorBatchDimIndex(num_dims(), data_format_);
    const int c_dim = GetTensorFeatureDimIndex(num_dims(), data_format_);

    OP_REQUIRES(
        ctx, (strides_[n_dim] == 1 && strides_[c_dim] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    OP_REQUIRES(ctx, dilations_.size() == num_dims(),
                errors::InvalidArgument("Dilations field must "
                                        "specify ",
                                        num_dims(), " dimensions"));
    OP_REQUIRES(
        ctx, dilations_[n_dim] == 1 && dilations_[c_dim] == 1,
        errors::Unimplemented("Current implementation does not support "
                              "dilations in the batch and depth dimensions."));
    for (int i = 0; i < num_spatial_dims_; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      OP_REQUIRES(ctx, dilations_[input_dim] >= 1,
                  errors::Unimplemented("Dilation values must be positive; ", i,
                                        "th spatial dimension had dilation ",
                                        dilations_[input_dim]));
    }

    const TensorShape activations_shape = ctx->InputShape(0);
    TensorShape filter_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(1, &filter_shape));
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    const TensorShape expanded_filter_shape =
        depthwise_ ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                   : filter_shape;

    // Reuse dimension computation logic from conv_grad_ops.cc.
    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(ctx,
                   ConvBackpropComputeDimensionsV2(
                       type_string(), num_spatial_dims_, activations_shape,
                       expanded_filter_shape, out_backprop_shape, dilations_,
                       strides_, padding_, data_format_, &dims));

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp activations = ctx->Input(0);
    xla::XlaOp gradients = ctx->Input(2);

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
    dnums.set_input_batch_dimension(c_dim);
    dnums.set_input_feature_dimension(n_dim);

    // The gradients become the RHS of the convolution.
    // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
    // where the batch becomes the input feature for the convolution.
    dnums.set_kernel_input_feature_dimension(n_dim);
    dnums.set_kernel_output_feature_dimension(c_dim);

    std::vector<std::pair<int64, int64>> padding(num_spatial_dims_);
    std::vector<int64> rhs_dilation(num_spatial_dims_);
    std::vector<int64> window_strides(num_spatial_dims_);
    std::vector<int64> ones(num_spatial_dims_, 1);

    // Tensorflow filter shape is [ H, W, ..., inC, outC ].
    for (int i = 0; i < num_spatial_dims_; ++i) {
      dnums.add_output_spatial_dimensions(i);
    }
    dnums.set_output_batch_dimension(num_spatial_dims_);
    dnums.set_output_feature_dimension(num_spatial_dims_ + 1);

    for (int i = 0; i < num_spatial_dims_; ++i) {
      int64 dim = GetTensorSpatialDimIndex(num_dims(), data_format_, i);
      dnums.add_input_spatial_dimensions(dim);
      dnums.add_kernel_spatial_dimensions(dim);

      // We will also need to pad the input with zeros such that after the
      // convolution, we get the right size for the filter.
      // The padded_in_rows should be such that when we convolve this with the
      // expanded_out_rows as a filter, we should get filter_rows back.
      //
      const int64 padded_in_size =
          dims.spatial_dims[i].expanded_output_size +
          (dims.spatial_dims[i].filter_size - 1) * dilations_[dim];

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
      const int64 pad_total = padded_in_size - dims.spatial_dims[i].input_size;

      // + For the VALID padding, we don't pad anything on the top/left side
      //   and pad the bottom/right side with the remaining space.
      // + For the SAME padding, we pad top/left side the same as bottom/right
      //   side.
      //
      // In addition, if the padded input size is smaller than the input size,
      // we need to ignore some training elements of the input. We do this by
      // applying negative padding on the right/bottom.
      const int64 pad_before =
          padding_ == Padding::SAME ? std::max<int64>(pad_total / 2, 0) : 0;

      padding[i] = {pad_before, pad_total - pad_before};
      rhs_dilation[i] = dims.spatial_dims[i].stride;
      window_strides[i] = dilations_[dim];
    }

    // Besides padding the input, we will also expand output_rows to
    //    expanded_out_rows = (output_rows - 1) * stride + 1
    // with zeros in between:
    //
    //      a . . . b . . . c . . . d . . . e
    //
    // This is done by specifying the window dilation factors in the
    // convolution HLO below.
    auto filter_backprop =
        xla::ConvGeneralDilated(activations, gradients, window_strides, padding,
                                /*lhs_dilation=*/ones, rhs_dilation, dnums);

    if (depthwise_) {
      filter_backprop = ContractFilterForDepthwiseBackprop(
          ctx, filter_shape, ctx->input_type(0), filter_backprop, b);
    }
    ctx->SetOutput(0, filter_backprop);
  }

 protected:
  const int num_spatial_dims_;
  const bool depthwise_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_ = FORMAT_NHWC;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ConvBackpropFilterOp);
};

class Conv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/false) {
  }
};
REGISTER_XLA_OP(
    Name("Conv2DBackpropFilter").CompileTimeConstInput("filter_sizes"),
    Conv2DBackpropFilterOp);

class Conv3DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/3, /*depthwise=*/false) {
  }
};
REGISTER_XLA_OP(
    Name("Conv3DBackpropFilterV2").CompileTimeConstInput("filter_sizes"),
    Conv3DBackpropFilterOp);

class DepthwiseConv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit DepthwiseConv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, /*num_spatial_dims=*/2, /*depthwise=*/true) {}
};
REGISTER_XLA_OP(Name("DepthwiseConv2dNativeBackpropFilter")
                    .CompileTimeConstInput("filter_sizes"),
                DepthwiseConv2DBackpropFilterOp);

}  // namespace
}  // namespace tensorflow
