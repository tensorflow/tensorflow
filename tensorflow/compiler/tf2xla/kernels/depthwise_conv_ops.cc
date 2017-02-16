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

// XLA-specific Ops for 2D depthwise convolution.

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
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
#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

// Name of the function to use as the implementation for depthwise 2D
// convolution.  Default is empty string; another possible value is
// "DummyDepthwiseConv2dKernel".
static const char kDepthwiseConv2dCustomFunc[] = "";

class DepthwiseConv2dNativeOp : public XlaOpKernel {
 public:
  explicit DepthwiseConv2dNativeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    // TODO(keveman): Refactor this (and other XLA OpKernel constructors) so
    // that they use a common implementation shared with non-XLA kernels.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES(ctx, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(ctx, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        ctx, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const TensorShape input_shape = ctx->InputShape(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const TensorShape filter_shape = ctx->InputShape(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(ctx, input_shape.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, filter_shape.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter_shape.DebugString()));

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = input_shape.dim_size(3);
    OP_REQUIRES(
        ctx, in_depth == filter_shape.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter_shape.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int64 depth_multiplier = filter_shape.dim_size(3);

    // The output depth is input depth x depth multiplier.
    const int64 out_depth = in_depth * depth_multiplier;

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows = input_shape.dim_size(1);
    const int64 filter_rows = filter_shape.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols = input_shape.dim_size(2);
    const int64 filter_cols = filter_shape.dim_size(1);

    // The first dimension for input is batch.
    const int64 batch = input_shape.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int32 stride = strides_[1];

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(ctx, GetWindowedOutputSize(input_rows, filter_rows, stride,
                                              padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(ctx, GetWindowedOutputSize(input_cols, filter_cols, stride,
                                              padding_, &out_cols, &pad_cols));
    TensorShape out_shape({batch, out_rows, out_cols, out_depth});
    OP_REQUIRES(
        ctx, out_shape.num_elements() <= 2147483647,
        errors::InvalidArgument("total number of outputs should be within the "
                                "range of int which is used in the GPU kernel",
                                in_depth, " vs ", filter_shape.dim_size(2)));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]

    VLOG(2) << "DepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; stride = " << stride << ", pad_rows = " << pad_rows
            << ", pad_cols = " << pad_cols << ", output: [" << batch << ", "
            << out_rows << ", " << out_cols << ", " << out_depth << "]";

    xla::ComputationBuilder& b = *ctx->builder();
    xla::ComputationDataHandle input = ctx->Input(0);
    xla::ComputationDataHandle filter = ctx->Input(1);
    xla::ComputationDataHandle output;

    const string custom_function_name = kDepthwiseConv2dCustomFunc;
    if (!custom_function_name.empty()) {
      xla::Shape xla_out_shape;
      OP_REQUIRES_OK(
          ctx, TensorShapeToXLAShape(input_type(0), out_shape, &xla_out_shape));

      // The custom function for depthwise should interpret its arguments
      // as follows :
      // func(T* output,
      //      const T* input, const T* filter,
      //      const int32* input_size, const int32* filter_size,
      //      const int32* output_size,
      //      int32 stride, int32 pad_rows, int32 pad_cols)
      //
      // where T is the type of Tensor that this kernel is registered for.
      // Note that the custom call op passes uses the following calling
      // convention:
      // func(void* output, void** inputs)
      //
      // Therefore the custom function should first construct the above
      // inputs by unparsing the second argument passed to it.
      output = b.CustomCall(
          custom_function_name,
          {input, filter,
           b.ConstantR1<int64>({batch, input_rows, input_cols, in_depth}),
           b.ConstantR1<int64>(
               {filter_rows, filter_cols, in_depth, depth_multiplier}),
           b.ConstantR1<int64>({batch, out_rows, out_cols, out_depth}),
           b.ConstantR0<int64>(stride), b.ConstantR0<int64>(pad_rows),
           b.ConstantR0<int64>(pad_cols)},
          xla_out_shape);
    } else {
      // These will be used to define the bounds of each slice.
      // Within the loop, the input_channel index will be modified.
      gtl::InlinedVector<int64, 4> filter_begin;
      gtl::InlinedVector<int64, 4> filter_limits;
      gtl::InlinedVector<int64, 4> input_begin;
      gtl::InlinedVector<int64, 4> input_limits;
      for (int i = 0; i < 4; ++i) {
        filter_begin.push_back(0);
        filter_limits.push_back(filter_shape.dim_size(i));
        input_begin.push_back(0);
        input_limits.push_back(input_shape.dim_size(i));
      }

      std::vector<int64> strides_for_tla{strides_[1], strides_[2]};

      xla::Padding xla_padding =
          (padding_ == VALID) ? xla::Padding::kValid : xla::Padding::kSame;

      xla::ConvolutionDimensionNumbers dims;
      dims.set_batch_dimension(0);
      dims.set_feature_dimension(3);
      dims.add_spatial_dimensions(1);
      dims.add_spatial_dimensions(2);

      // TF filter shape is [ H, W, inC, outC ]
      dims.add_kernel_spatial_dimensions(0);
      dims.add_kernel_spatial_dimensions(1);
      dims.set_kernel_input_feature_dimension(2);
      dims.set_kernel_output_feature_dimension(3);

      // Create one convolution for each input channel
      std::vector<xla::ComputationDataHandle> convs;
      for (int i = 0; i < in_depth; ++i) {
        filter_begin[2] = i;
        filter_limits[2] = i + 1;
        input_begin[3] = i;
        input_limits[3] = i + 1;

        xla::ComputationDataHandle filter_slice =
            b.Slice(filter, filter_begin, filter_limits);
        xla::ComputationDataHandle input_slice =
            b.Slice(input, input_begin, input_limits);
        convs.push_back(b.ConvWithGeneralDimensions(
            input_slice, filter_slice, strides_for_tla, xla_padding, dims));
      }
      // Concatenate the per-channel convolutions along the depth dimension.
      output = b.ConcatInDim(convs, 3);
    }

    ctx->SetOutput(0, output);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeOp);
};

REGISTER_XLA_OP("DepthwiseConv2dNative", DepthwiseConv2dNativeOp);

}  // namespace
}  // namespace tensorflow
