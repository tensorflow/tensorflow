/* Copyright 2015 Google Inc. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchDepthwiseConvOp;

template <typename T>
struct LaunchDepthwiseConvOp<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* input, const T* filter, T* output) {
    // Naive for loop as a reference point without concerns about performance.
    // Expected to be replaced later.
    // TODO(andydavis): replace this with an optimized version
    for (int b = 0; b < args.batch; ++b) {
      for (int out_r = 0; out_r < args.out_rows; ++out_r) {
        for (int out_c = 0; out_c < args.out_cols; ++out_c) {
          for (int out_d = 0; out_d < args.out_depth; ++out_d) {
            T sum = 0;
            const int in_r_start = out_r * args.stride - args.pad_rows;
            const int in_c_start = out_c * args.stride - args.pad_cols;
            const int in_d = out_d / args.depth_multiplier;
            const int filter_dm = out_d % args.depth_multiplier;

            for (int f_r = 0; f_r < args.filter_rows; ++f_r) {
              for (int f_c = 0; f_c < args.filter_cols; ++f_c) {
                int in_r = in_r_start + f_r;
                int in_c = in_c_start + f_c;

                if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
                    in_c < args.in_cols) {
                  int input_offset =
                      in_d +
                      args.in_depth *
                          (in_c + args.in_cols * (in_r + args.in_rows * b));
                  int filter_offset =
                      filter_dm +
                      args.depth_multiplier *
                          (in_d +
                           args.in_depth * (f_c + args.filter_cols * f_r));
                  sum += input[input_offset] * filter[filter_offset];
                }
              }
            }

            int output_offset =
                out_d +
                args.out_depth *
                    (out_c + args.out_cols * (out_r + args.out_rows * b));
            output[output_offset] = sum;
          }
        }
      }
    }
  }
};

#if GOOGLE_CUDA

template <typename T>
struct DepthwiseConv2dGPULaunch {
  void Run(const GPUDevice& d, const DepthwiseArgs args, const T* input,
           const T* filter, T* output);
};

template <typename T>
struct LaunchDepthwiseConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const T* input, const T* filter, T* output) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    DepthwiseConv2dGPULaunch<T>().Run(d, args, input, filter, output);
    auto stream = ctx->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(ctx, stream->ok(),
                errors::Internal("Launch of gpu kernel for SplitOp failed"));
  }
};

#endif

template <typename Device, typename T>
class DepthwiseConv2dNativeOp : public BinaryOp<T> {
 public:
  explicit DepthwiseConv2dNativeOp(OpKernelConstruction* context)
      : BinaryOp<T>(context) {
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
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);
    auto input_ptr = input.template flat<T>().data();

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);
    auto filter_ptr = filter.template flat<T>().data();

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int32 in_depth = input.dim_size(3);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int32 depth_multiplier = filter.dim_size(3);

    // The output depth is input depth x depth multipler
    const int32 out_depth = in_depth * depth_multiplier;

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int32 input_rows = input.dim_size(1);
    const int32 filter_rows = filter.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int32 input_cols = input.dim_size(2);
    const int32 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int32 stride = strides_[1];

    int32 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   Get2dOutputSize(input_rows, input_cols, filter_rows,
                                   filter_cols, stride, stride, padding_,
                                   &out_rows, &out_cols, &pad_rows, &pad_cols));
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
    auto output_ptr = output->template flat<T>().data();

    DepthwiseArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = stride;
    args.pad_rows = pad_rows;
    args.pad_cols = pad_cols;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    VLOG(2) << "DepthwiseConv2dNative: "
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
    LaunchDepthwiseConvOp<Device, T>::launch(context, args, input_ptr,
                                             filter_ptr, output_ptr);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeOp);
};

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    DepthwiseConv2dNativeOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        DepthwiseConv2dNativeOp<CPUDevice, double>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    DepthwiseConv2dNativeOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        DepthwiseConv2dNativeOp<GPUDevice, double>);
#endif

}  // namespace tensorflow
