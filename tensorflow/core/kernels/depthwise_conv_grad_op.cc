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
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Gradient operations for depthwise convolution.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Common code between the two backward pass kernels: verifies that the
// dimensions all match and extract the padded rows and columns.
#define EXTRACT_AND_VERIFY_DIMENSIONS(label)                                   \
  const Tensor& out_backprop = context->input(2);                              \
  OP_REQUIRES(                                                                 \
      context, input_shape.dims() == 4,                                        \
      errors::InvalidArgument(label, ": input must be 4-dimensional"));        \
  OP_REQUIRES(                                                                 \
      context, filter_shape.dims() == 4,                                       \
      errors::InvalidArgument(label, ": filter must be 4-dimensional"));       \
  OP_REQUIRES(                                                                 \
      context, out_backprop.dims() == 4,                                       \
      errors::InvalidArgument(label, ": out_backprop must be 4-dimensional")); \
  const int64 batch = input_shape.dim_size(0);                                 \
  OP_REQUIRES(                                                                 \
      context, batch == out_backprop.dim_size(0),                              \
      errors::InvalidArgument(                                                 \
          label, ": input and out_backprop must have the same batch size"));   \
  const int64 input_rows = input_shape.dim_size(1);                            \
  const int64 input_cols = input_shape.dim_size(2);                            \
  const int64 filter_rows = filter_shape.dim_size(0);                          \
  const int64 filter_cols = filter_shape.dim_size(1);                          \
  const int64 output_rows = out_backprop.dim_size(1);                          \
  const int64 output_cols = out_backprop.dim_size(2);                          \
  const int64 in_depth = input_shape.dim_size(3);                              \
  OP_REQUIRES(context, in_depth == filter_shape.dim_size(2),                   \
              errors::InvalidArgument(                                         \
                  label, ": input and filter must have the same in_depth"));   \
  const int64 depth_multiplier = filter_shape.dim_size(3);                     \
  const int64 out_depth = out_backprop.dim_size(3);                            \
  OP_REQUIRES(                                                                 \
      context, (depth_multiplier * in_depth) == out_depth,                     \
      errors::InvalidArgument(                                                 \
          label, ": depth_multiplier * in_depth not equal to out_depth"));     \
  const auto stride = strides_[1];                                             \
  int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;                  \
  if (filter_cols == filter_rows && filter_rows == 1 && stride == 1) {         \
    out_rows = input_rows;                                                     \
    out_cols = input_cols;                                                     \
  } else {                                                                     \
    OP_REQUIRES_OK(                                                            \
        context, Get2dOutputSize(input_rows, input_cols, filter_rows,          \
                                 filter_cols, stride, stride, padding_,        \
                                 &out_rows, &out_cols, &pad_rows, &pad_cols)); \
  }                                                                            \
  OP_REQUIRES(                                                                 \
      context, output_rows == out_rows,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of rows of out_backprop doesn't match computed: ",  \
          "actual = ", output_rows, ", computed = ", out_rows));               \
  OP_REQUIRES(                                                                 \
      context, output_cols == out_cols,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of cols of out_backprop doesn't match computed: ",  \
          "actual = ", output_cols, ", computed = ", out_cols));               \
  DepthwiseArgs args;                                                          \
  args.batch = batch;                                                          \
  args.in_rows = input_rows;                                                   \
  args.in_cols = input_cols;                                                   \
  args.in_depth = in_depth;                                                    \
  args.filter_rows = filter_rows;                                              \
  args.filter_cols = filter_cols;                                              \
  args.depth_multiplier = depth_multiplier;                                    \
  args.stride = stride;                                                        \
  args.pad_rows = pad_rows;                                                    \
  args.pad_cols = pad_cols;                                                    \
  args.out_rows = out_rows;                                                    \
  args.out_cols = out_cols;                                                    \
  args.out_depth = out_depth;                                                  \
  VLOG(2) << "DepthwiseConv2d: " << label << " Input: [" << batch << ", "      \
          << input_rows << ", " << input_cols << ", " << in_depth              \
          << "]; Filter: [" << filter_rows << ", " << filter_cols << ", "      \
          << in_depth << ", " << depth_multiplier << "]; stride = " << stride  \
          << ", pad_rows = " << pad_rows << ", pad_cols = " << pad_cols        \
          << ", output: [" << batch << ", " << out_rows << ", " << out_cols    \
          << ", " << out_depth << "]";

// Kernels to compute the input backprop for depthwise convolution.
template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropInputOp;

template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* out_backprop, const T* filter, T* in_backprop) {
    // Naive for loop as a reference point without concerns about performance.
    // Expected to be replaced later.
    // TODO(andydavis): replace this with an optimized version
    for (int b = 0; b < args.batch; ++b) {
      for (int in_r = 0; in_r < args.in_rows; ++in_r) {
        for (int in_c = 0; in_c < args.in_cols; ++in_c) {
          for (int in_d = 0; in_d < args.in_depth; ++in_d) {
            T sum = 0;
            const int stride = args.stride;
            const int out_d_start = in_d * args.depth_multiplier;
            const int out_d_end = out_d_start + args.depth_multiplier;

            for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
              const int out_r_start = std::max(
                  0,
                  (in_r - args.filter_rows + args.pad_rows + stride) / stride);
              const int out_r_end =
                  std::min(args.out_rows - 1, (in_r + args.pad_rows) / stride);

              for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
                const int out_c_start = std::max(
                    0, (in_c - args.filter_cols + args.pad_cols + stride) /
                           stride);
                const int out_c_end = std::min(args.out_cols - 1,
                                               (in_c + args.pad_cols) / stride);

                for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
                  int f_r = in_r + args.pad_rows - out_r * stride;
                  int f_c = in_c + args.pad_cols - out_c * stride;
                  int filter_dm = out_d - out_d_start;
                  int out_backprop_offset =
                      out_d +
                      args.out_depth *
                          (out_c + args.out_cols * (out_r + args.out_rows * b));
                  int filter_offset =
                      filter_dm +
                      args.depth_multiplier *
                          (in_d +
                           args.in_depth * (f_c + args.filter_cols * f_r));
                  sum +=
                      out_backprop[out_backprop_offset] * filter[filter_offset];
                }
              }
            }

            int in_backprop_offset =
                in_d +
                args.in_depth *
                    (in_c + args.in_cols * (in_r + args.in_rows * b));
            in_backprop[in_backprop_offset] = sum;
          }
        }
      }
    }
  }
};

#if GOOGLE_CUDA

template <typename T>
struct DepthwiseConv2dBackpropInputGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args,
                  const T* out_backprop, const T* filter, T* in_backprop);
};

template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const T* out_backprop, const T* filter, T* in_backprop) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    DepthwiseConv2dBackpropInputGPULaunch<T>().Run(d, args, out_backprop,
                                                   filter, in_backprop);
    auto stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream->ok(), errors::Internal("Launch of gpu kernel for "
                                                    "DepthwiseConv2dBackpropInp"
                                                    "utGPULaunch failed"));
  }
};

#endif  // GOOGLE_CUDA

// Kernel to compute the input backprop for depthwise convolution.
template <typename Device, class T>
class DepthwiseConv2dNativeBackpropInputOp : public OpKernel {
 public:
  explicit DepthwiseConv2dNativeBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
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
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    const int32* in_sizes_data = input_sizes.template flat<int32>().data();
    for (int i = 0; i < input_sizes.NumElements(); ++i) {
      OP_REQUIRES(context, in_sizes_data[i] >= 0,
                  errors::InvalidArgument("Dimension ", i,
                                          " of input_sizes must be >= 0"));
      input_shape.AddDim(in_sizes_data[i]);
    }
    const TensorShape& filter_shape = filter.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("DepthwiseConv2DBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    auto out_backprop_ptr = out_backprop.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto in_backprop_ptr = in_backprop->template flat<T>().data();
    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }
    LaunchDepthwiseConvBackpropInputOp<Device, T>::launch(
        context, args, out_backprop_ptr, filter_ptr, in_backprop_ptr);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeBackpropInputOp);
};

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNativeBackpropInput")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        DepthwiseConv2dNativeBackpropInputOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropInput")
        .Device(DEVICE_CPU)
        .TypeConstraint<double>("T"),
    DepthwiseConv2dNativeBackpropInputOp<CPUDevice, double>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNativeBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("input_sizes"),
                        DepthwiseConv2dNativeBackpropInputOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropInput")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T")
        .HostMemory("input_sizes"),
    DepthwiseConv2dNativeBackpropInputOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

// Kernels to compute the gradients of the filters for depthwise convolution.
template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropFilterOp;

template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* out_backprop, const T* input,
                     T* filter_backprop) {
    int num_filter_backprop = args.filter_rows * args.filter_cols *
                              args.in_depth * args.depth_multiplier;
    memset(filter_backprop, 0, num_filter_backprop * sizeof(T));

    // Naive for loop as a reference point without concerns about performance.
    // Expected to be replaced later.
    // TODO(andydavis): replace this with an optimized version
    for (int b = 0; b < args.batch; ++b) {
      for (int out_r = 0; out_r < args.out_rows; ++out_r) {
        for (int out_c = 0; out_c < args.out_cols; ++out_c) {
          for (int out_d = 0; out_d < args.out_depth; ++out_d) {
            const int in_d = out_d / args.depth_multiplier;
            const int dm = out_d % args.depth_multiplier;
            const int in_r_start = out_r * args.stride - args.pad_rows;
            const int in_c_start = out_c * args.stride - args.pad_cols;

            for (int f_r = 0; f_r < args.filter_rows; ++f_r) {
              for (int f_c = 0; f_c < args.filter_cols; ++f_c) {
                const int in_r = in_r_start + f_r;
                const int in_c = in_c_start + f_c;

                if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
                    in_c < args.in_cols) {
                  int out_backprop_offset =
                      out_d +
                      args.out_depth *
                          (out_c + args.out_cols * (out_r + args.out_rows * b));
                  int input_offset =
                      in_d +
                      args.in_depth *
                          (in_c + args.in_cols * (in_r + args.in_rows * b));
                  int filter_backprop_offset =
                      dm +
                      args.depth_multiplier *
                          (in_d +
                           args.in_depth * (f_c + args.filter_cols * f_r));
                  filter_backprop[filter_backprop_offset] +=
                      input[input_offset] * out_backprop[out_backprop_offset];
                }
              }
            }
          }
        }
      }
    }
  }
};

#if GOOGLE_CUDA

template <typename T>
struct DepthwiseConv2dBackpropFilterGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args,
                  const T* out_backprop, const T* input, T* filter_backprop);
};

template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const T* out_backprop, const T* input,
                     T* filter_backprop) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    auto stream = ctx->op_device_context()->stream();

    // Initialize the results to 0.
    int num_filter_backprop =
        args.filter_rows * args.filter_cols * args.out_depth;
    perftools::gputools::DeviceMemoryBase filter_bp_ptr(filter_backprop,
                                                        num_filter_backprop);
    stream->ThenMemset32(&filter_bp_ptr, 0, num_filter_backprop * sizeof(T));

    DepthwiseConv2dBackpropFilterGPULaunch<T>().Run(d, args, out_backprop,
                                                    input, filter_backprop);
    OP_REQUIRES(ctx, stream->ok(), errors::Internal("Launch of gpu kernel for "
                                                    "DepthwiseConv2dBackpropFil"
                                                    "terGPULaunch failed"));
  }
};

#endif  // GOOGLE_CUDA

// Kernel to compute the input backprop for depthwise convolution.
template <typename Device, class T>
class DepthwiseConv2dNativeBackpropFilterOp : public OpKernel {
 public:
  explicit DepthwiseConv2dNativeBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
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
    const Tensor& filter_sizes = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    const int32* filter_sizes_data = filter_sizes.template flat<int32>().data();
    for (int i = 0; i < filter_sizes.NumElements(); ++i) {
      OP_REQUIRES(context, filter_sizes_data[i] >= 0,
                  errors::InvalidArgument("Dimension ", i,
                                          " of filter_sizes must be >= 0"));
      filter_shape.AddDim(filter_sizes_data[i]);
    }
    const TensorShape& input_shape = input.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("DepthwiseConv2DBackpropFilter");
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    auto out_backprop_ptr = out_backprop.template flat<T>().data();
    auto input_ptr = input.template flat<T>().data();
    auto filter_backprop_ptr = filter_backprop->template flat<T>().data();
    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }
    LaunchDepthwiseConvBackpropFilterOp<Device, T>::launch(
        context, args, out_backprop_ptr, input_ptr, filter_backprop_ptr);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeBackpropFilterOp);
};

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
    DepthwiseConv2dNativeBackpropFilterOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_CPU)
        .TypeConstraint<double>("T"),
    DepthwiseConv2dNativeBackpropFilterOp<CPUDevice, double>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .HostMemory("filter_sizes"),
    DepthwiseConv2dNativeBackpropFilterOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNativeBackpropFilter")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T")
        .HostMemory("filter_sizes"),
    DepthwiseConv2dNativeBackpropFilterOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
