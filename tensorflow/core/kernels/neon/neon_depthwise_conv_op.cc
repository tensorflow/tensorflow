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

#include <algorithm>
#include <cmath>
#include <type_traits>

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/neon/depthwiseconv_float.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// A version of tensorflow/core/kernels/depthwise_conv_op.cc that
// uses the neon intrinsics.
class NeonDepthwiseConv2dNativeOp : public BinaryOp<float> {
 public:
  explicit NeonDepthwiseConv2dNativeOp(OpKernelConstruction* context)
      : BinaryOp<float>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    const int32 in_depth = input.dim_size(3);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));
    const int32 batch = input.dim_size(0);
    const int32 input_rows = input.dim_size(1);
    const int32 input_cols = input.dim_size(2);

    const int32 filter_rows = filter.dim_size(0);
    const int32 filter_cols = filter.dim_size(1);
    const int32 depth_multiplier = filter.dim_size(3);

    const int32 out_depth = in_depth * depth_multiplier;

    const int32 stride = strides_[1];

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape({batch, out_rows, out_cols, out_depth});
    OP_REQUIRES(
        context, out_shape.num_elements() <= 2147483647,
        errors::InvalidArgument("total number of outputs should be within the "
                                "range of int which is used in the GPU kernel",
                                in_depth, " vs ", filter.dim_size(2)));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "NeonDepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; stride = " << stride << ", pad_rows = " << pad_rows
            << ", pad_cols = " << pad_cols << ", output: [" << batch << ", "
            << out_rows << ", " << out_cols << ", " << out_depth << "]";

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    const float* input_ptr = input.template flat<float>().data();
    const float* filter_ptr = filter.template flat<float>().data();
    float* output_ptr = output->template flat<float>().data();

    auto input_neon_dims = ToNeonDims(input.shape());
    auto filter_neon_dims = FilterToNeonDims(filter.shape());
    auto bias_neon_dims = BiasNeonDims(filter.shape());

    int64 bias_size = bias_neon_dims.sizes[0];
    float* bias_ptr = static_cast<float*>(port::AlignedMalloc(
        bias_size * sizeof(float), Allocator::kAllocatorAlignment));
    memset(bias_ptr, 0, bias_size * sizeof(float));

    neon::DepthwiseConv<neon::FusedActivationFunctionType::kNone>(
        input_ptr, input_neon_dims, filter_ptr, filter_neon_dims, bias_ptr,
        bias_neon_dims, stride, pad_cols, pad_rows, depth_multiplier,
        output_ptr, ToNeonDims(out_shape));

    port::AlignedFree(bias_ptr);
  }

 private:
  void SetNeonDimStrides(neon::Dims<4>* d) {
    int64 stride = 1;
    for (int i = 0; i < 4; ++i) {
      d->strides[i] = stride;
      stride *= d->sizes[i];
    }
  }

  neon::Dims<4> ToNeonDims(const TensorShape& input) {
    // Dims in the neon kernels are channel, x, y, batch order.
    neon::Dims<4> result;
    result.sizes[0] = input.dim_size(3);
    result.sizes[1] = input.dim_size(2);
    result.sizes[2] = input.dim_size(1);
    result.sizes[3] = input.dim_size(0);
    SetNeonDimStrides(&result);
    return result;
  }

  neon::Dims<4> FilterToNeonDims(const TensorShape& filter) {
    // Dims in the neon kernels are channel, x, y, batch order.
    neon::Dims<4> result;
    result.sizes[0] = filter.dim_size(2) * filter.dim_size(3);
    result.sizes[1] = filter.dim_size(1);
    result.sizes[2] = filter.dim_size(0);
    result.sizes[3] = 1;
    SetNeonDimStrides(&result);

    return result;
  }

  neon::Dims<4> BiasNeonDims(const TensorShape& filter) {
    // Dims in the neon kernels are channel, x, y, batch order.
    // Bias has only output channel set.
    neon::Dims<4> result;
    result.sizes[0] =
        filter.dim_size(2) * filter.dim_size(3);  // output channels
    result.sizes[1] = 1;
    result.sizes[2] = 1;
    result.sizes[3] = 1;
    SetNeonDimStrides(&result);

    return result;
  }

  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(NeonDepthwiseConv2dNativeOp);
};

#define REGISTER_CPU_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<float>("T") \
                              .Label("neon"),             \
                          NeonDepthwiseConv2dNativeOp);

TF_CALL_float(REGISTER_CPU_KERNEL);

}  // namespace tensorflow
