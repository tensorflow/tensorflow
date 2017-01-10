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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
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

class Conv2DOp : public XlaOpKernel {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(ctx, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        ctx, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const TensorShape filter_shape = ctx->InputShape(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(ctx, input_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, filter_shape.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter_shape.DebugString()));

    // The 'C' dimension for input is in_depth. It must be the same as
    // the filter's in_depth.
    const int64 in_depth = GetTensorDim(input_shape, data_format_, 'C');
    OP_REQUIRES(
        ctx, in_depth == filter_shape.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter_shape.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int64 out_depth = filter_shape.dim_size(3);

    // The 'H' dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows = GetTensorDim(input_shape, data_format_, 'H');
    const int64 filter_rows = filter_shape.dim_size(0);

    // The 'W' dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols = GetTensorDim(input_shape, data_format_, 'W');
    const int64 filter_cols = filter_shape.dim_size(1);

    // For now we take the stride from the H and W dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(ctx,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(ctx,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));

    VLOG(2) << "Conv2D: in_depth = " << in_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", out_depth = " << out_depth;

    xla::ConvolutionDimensionNumbers dims;
    dims.set_batch_dimension(GetTensorDimIndex<2>(data_format_, 'N'));
    dims.set_feature_dimension(GetTensorDimIndex<2>(data_format_, 'C'));
    dims.add_spatial_dimensions(GetTensorDimIndex<2>(data_format_, 'H'));
    dims.add_spatial_dimensions(GetTensorDimIndex<2>(data_format_, 'W'));

    // TF filter shape is [ H, W, inC, outC ]
    dims.add_kernel_spatial_dimensions(0);
    dims.add_kernel_spatial_dimensions(1);
    dims.set_kernel_input_feature_dimension(2);
    dims.set_kernel_output_feature_dimension(3);

    std::vector<int64> window_strides = {stride_rows, stride_cols};
    xla::Padding xla_padding =
        (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

    xla::ComputationDataHandle conv = ctx->builder()->ConvWithGeneralDimensions(
        ctx->Input(0), ctx->Input(1), window_strides, xla_padding, dims);
    ctx->SetOutput(0, conv);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DOp);
};

REGISTER_XLA_OP("Conv2D", Conv2DOp);

// Backprop for input.
class Conv2DBackpropInputOp : public XlaOpKernel {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES(ctx, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        ctx, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape input_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &input_shape));

    const TensorShape filter_shape = ctx->InputShape(1);
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // Reuse dimension computation logic from conv_grad_ops.cc.
    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, Conv2DBackpropComputeDimensions(
                 "Conv2DBackpropInput", input_shape, filter_shape,
                 out_backprop_shape, strides_, padding_, data_format_, &dims));

    auto filter = ctx->Input(1);
    auto out_backprop = ctx->Input(2);

    // The input gradients are computed by a convolution of the output
    // gradients and the filter, with some appropriate padding. See the
    // comment at the top of conv_grad_ops.h for details.

    xla::ConvolutionDimensionNumbers dnums;
    dnums.set_batch_dimension(GetTensorDimIndex(data_format_, 'N'));
    dnums.add_spatial_dimensions(GetTensorDimIndex(data_format_, 'H'));
    dnums.add_spatial_dimensions(GetTensorDimIndex(data_format_, 'W'));
    dnums.set_feature_dimension(GetTensorDimIndex(data_format_, 'C'));

    // TF filter shape is [ H, W, inC, outC ]
    // Transpose the input and output features for computing the gradient.
    dnums.add_kernel_spatial_dimensions(0);
    dnums.add_kernel_spatial_dimensions(1);
    dnums.set_kernel_input_feature_dimension(3);
    dnums.set_kernel_output_feature_dimension(2);

    // Mirror the filter in the spatial dimensions.
    xla::ComputationDataHandle mirrored_weights =
        ctx->builder()->Rev(filter, {dnums.kernel_spatial_dimensions(0),
                                     dnums.kernel_spatial_dimensions(1)});

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    xla::ComputationDataHandle in_backprop = ctx->builder()->ConvGeneralDilated(
        out_backprop, mirrored_weights, /*window_strides=*/{1, 1},
        /*padding=*/{{dims.rows.pad_before, dims.rows.pad_after},
                     {dims.cols.pad_before, dims.cols.pad_after}},
        /*lhs_dilation=*/{dims.rows.stride, dims.cols.stride},
        /*rhs_dilation=*/{1, 1}, dnums);

    ctx->SetOutput(0, in_backprop);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DBackpropInputOp);
};

class Conv2DBackpropFilterOp : public XlaOpKernel {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    OP_REQUIRES(ctx, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        ctx, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape activations_shape = ctx->InputShape(0);
    TensorShape filter_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(1, &filter_shape));
    const TensorShape out_backprop_shape = ctx->InputShape(2);

    // Reuse dimension computation logic from conv_grad_ops.cc.
    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, Conv2DBackpropComputeDimensions(
                 "Conv2DBackpropFilter", activations_shape, filter_shape,
                 out_backprop_shape, strides_, padding_, data_format_, &dims));

    xla::ComputationDataHandle activations = ctx->Input(0);
    xla::ComputationDataHandle gradients = ctx->Input(2);

    // The filter gradients are computed by a convolution of the input
    // activations and the output gradients, with some appropriate padding.
    // See the comment at the top of conv_grad_ops.h for details.

    xla::ConvolutionDimensionNumbers dnums;

    // The activations (inputs) form the LHS of the convolution.
    // Activations have shape: [batch, in_rows, in_cols, in_depth]
    // For the gradient computation, we flip the roles of the batch and
    // feature dimensions.
    // Each spatial entry has size in_depth * batch
    const int n_dim = GetTensorDimIndex(data_format_, 'N');
    const int h_dim = GetTensorDimIndex(data_format_, 'H');
    const int w_dim = GetTensorDimIndex(data_format_, 'W');
    const int c_dim = GetTensorDimIndex(data_format_, 'C');

    // Swap n_dim and c_dim in the activations.
    dnums.set_batch_dimension(c_dim);
    dnums.add_spatial_dimensions(h_dim);
    dnums.add_spatial_dimensions(w_dim);
    dnums.set_feature_dimension(n_dim);

    // The gradients become the RHS of the convolution.
    // The gradients have shape [batch, out_rows, out_cols, out_depth] where
    // the batch becomes the input feature for the convolution.
    dnums.add_kernel_spatial_dimensions(h_dim);
    dnums.add_kernel_spatial_dimensions(w_dim);
    dnums.set_kernel_input_feature_dimension(n_dim);
    dnums.set_kernel_output_feature_dimension(c_dim);

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.
    //
    const int padded_in_rows =
        dims.rows.expanded_output_size + dims.rows.filter_size - 1;
    const int padded_in_cols =
        dims.cols.expanded_output_size + dims.cols.filter_size - 1;

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
    const int total_pad_in_rows = padded_in_rows - dims.rows.input_size;
    const int total_pad_in_cols = padded_in_cols - dims.cols.input_size;

    // + For the VALID padding, we don't pad anything on the top/left side
    //   and pad the bottom/right side with the remaining space.
    // + For the SAME padding, we pad top/left side the same as bottom/right
    //   side.
    //
    // In addition, if the padded input size is smaller than the input size,
    // we need to ignore some training elements of the input. We do this by
    // applying negative padding on the right/bottom.
    const int top_pad_in_rows =
        (total_pad_in_rows > 0 && padding_ == Padding::SAME)
            ? total_pad_in_rows / 2
            : 0;
    const int left_pad_in_cols =
        (total_pad_in_cols > 0 && padding_ == Padding::SAME)
            ? total_pad_in_cols / 2
            : 0;

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
        /*window_strides=*/{1, 1},
        /*padding=*/{{top_pad_in_rows, total_pad_in_rows - top_pad_in_rows},
                     {left_pad_in_cols, total_pad_in_cols - left_pad_in_cols}},
        /*lhs_dilation=*/{1, 1},
        /*rhs_dilation=*/{dims.rows.stride, dims.cols.stride}, dnums);

    // The layout of filter_backprop will match the layout of
    // padded_activations
    // and so will have layout: [out_feature, h, w, in_feature]
    // Tensorflow filter shape is [ H, W, inC, outC ], so we transpose the
    // output.
    xla::ComputationDataHandle filter_backprop_reshaped =
        ctx->builder()->Transpose(filter_backprop,
                                  {h_dim, w_dim, c_dim, n_dim});
    ctx->SetOutput(0, filter_backprop_reshaped);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DBackpropFilterOp);
};

REGISTER_XLA_OP("Conv2DBackpropInput", Conv2DBackpropInputOp);
REGISTER_XLA_OP("Conv2DBackpropFilter", Conv2DBackpropFilterOp);

}  // namespace
}  // namespace tensorflow
