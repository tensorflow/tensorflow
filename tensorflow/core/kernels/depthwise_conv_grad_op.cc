/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;                \
  OP_REQUIRES_OK(context,                                                      \
                 GetWindowedOutputSize(input_rows, filter_rows, stride,        \
                                       padding_, &out_rows, &pad_rows));       \
  OP_REQUIRES_OK(context,                                                      \
                 GetWindowedOutputSize(input_cols, filter_cols, stride,        \
                                       padding_, &out_cols, &pad_cols));       \
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

// Copies data from local region in 'out_backprop' into 'buffer'.
// The local region coordinates are calculated as the set of output points which
// used the input point ('in_r', 'in_'c') as input during the forward pass.
// Rather than spatially reversing the filter, the input is reversed during
// the copy. The copied data is padded to vector register-width boundaries so
// that it is aligned for efficient traversal and vector multiply-add by the
// depthwise input kernel.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//
//   'out_backprop': [batch, out_rows, out_cols, out_depth]
//
//     [a00, a01, a10, a11] [a20, a21, b00, b01]
//     [b10, b11, b20, b21] [...]
//     [e00, e01, e10, e11] [e20, e21, f00, f01]
//     [f10, f11, f20, f21] [...]
//
//   'buffer' (register boundaries shown):
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
template <typename T>
static void CopyOutputBackpropRegion(const DepthwiseArgs& args,
                                     const int64 padded_filter_inner_dim_size,
                                     const int64 in_r, const int64 in_c,
                                     const T* out_backprop, T* buffer) {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

  const int64 stride = args.stride;
  const int64 filter_rows = args.filter_rows;
  const int64 filter_cols = args.filter_cols;
  const int64 pad_rows = args.pad_rows;
  const int64 pad_cols = args.pad_cols;
  const int64 out_rows = args.out_rows;
  const int64 out_cols = args.out_cols;

  // Calculate the output spatial region which used point (in_r, in_c) as input.
  const int64 out_r_start = std::max(
      static_cast<int64>(0), (in_r - filter_rows + pad_rows + stride) / stride);
  const int64 out_r_end = std::min(out_rows - 1, (in_r + pad_rows) / stride);
  const int64 out_c_start = std::max(
      static_cast<int64>(0), (in_c - filter_cols + pad_cols + stride) / stride);
  const int64 out_c_end = std::min(out_cols - 1, (in_c + pad_cols) / stride);

  // Zero-pad 'buffer' if output region is smaller than filter spatial size.
  const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
  if ((out_r_end - out_r_start + 1) < args.filter_rows ||
      (out_c_end - out_c_start + 1) < args.filter_cols) {
    memset(buffer, 0,
           filter_spatial_size * padded_filter_inner_dim_size * sizeof(T));
  }

  // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
  const int64 vectorized_size = (args.out_depth / kPacketSize) * kPacketSize;
  const int64 scalar_size = args.out_depth % kPacketSize;
  const int64 pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;

  for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
    const int64 f_r = in_r + pad_rows - out_r * stride;
    for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
      const int64 f_c = in_c + pad_cols - out_c * stride;
      const int64 buf_base =
          (f_r * filter_cols + f_c) * padded_filter_inner_dim_size;
      // Calculate index into 'out_backprop' for coordinate (out_r, out_c).
      auto* out_bprop =
          out_backprop + (out_r * args.out_cols + out_c) * args.out_depth;

      // Copy vectorized portion of inner dimension into 'buffer'.
      for (int64 d = 0; d < vectorized_size; d += kPacketSize) {
        auto v = Eigen::internal::ploadu<Packet>(out_bprop + d);
        Eigen::internal::pstoreu<T>(buffer + buf_base + d, v);
      }
      // Copy scalar portion of out_bprop to 'buffer'
      for (int64 d = 0; d < scalar_size; ++d) {
        buffer[buf_base + vectorized_size + d] = out_bprop[vectorized_size + d];
      }
      // Pad to vector-register width (if needed).
      for (int64 d = 0; d < pad_size; ++d) {
        buffer[buf_base + vectorized_size + scalar_size + d] = 0;
      }
    }
  }
}

// Computes the vectorized product of 'buffer' and 'filter' and stores
// result in 'output' at location computed from 'in_r' and 'in_c'.
// If depth_multiplier is > 1, the intermediate output is reduced along
// the depth_multiplier dimension.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   'buffer' [rows, cols, in_depth, depth_multiplier]
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q00, q01, q10, q11] = ([f00, f01, f10, f11] x [u0, v0, w0, x0]) +
//                            ([e00, e01, e10, e11] x [u1, v1, w1, x1]) +
//                            ([b00, b01, b10, b11] x [u2, v2, w2, x2]) +
//                            ([a00, a01, a10, a11] x [u3, v3, w3, x3])
//
//   Reduction step along depth-multiplier dimension:
//
//     [q00, q01, q10, q11] [q20, q21, 0, 0] -> [r0, r1, r2, 0]
//

template <typename T>
static void ComputeBackpropInput(const DepthwiseArgs& args,
                                 const int64 padded_filter_inner_dim_size,
                                 const int64 in_r, const int64 in_c,
                                 const T* filter, const T* buffer,
                                 T* out_buffer, T* output) {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

  const int64 in_depth = args.in_depth;
  const int64 depth_multiplier = args.depth_multiplier;
  const int64 out_depth = args.out_depth;
  const int64 filter_spatial_size = args.filter_rows * args.filter_cols;

  // Calculate vectorized and scalar lengths of 'out_depth'.
  const int64 output_vectorized_size = (out_depth / kPacketSize) * kPacketSize;
  const int64 output_scalar_size = out_depth % kPacketSize;

  // Calculate base index at which to begin writing output.
  const int64 base_output_index = (in_r * args.in_cols + in_c) * in_depth;

  // Calculate vectorized and scalar lengths for 'depth_multiplier'. This is
  // used to efficiently reduce output when 'depth_multiplier' > kPacketSize.
  const int64 dm_vectorized_size =
      (depth_multiplier / kPacketSize) * kPacketSize;
  const int64 dm_scalar_size = depth_multiplier % kPacketSize;

  for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
    // Reset accumulator.
    auto vaccum = Eigen::internal::pset1<Packet>(0);
    for (int j = 0; j < filter_spatial_size; ++j) {
      // Calculate index.
      const int64 index = i + j * padded_filter_inner_dim_size;
      // Load filter.
      const auto filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      // Load input.
      const auto data_block = Eigen::internal::ploadu<Packet>(buffer + index);
      // Vector multiply-add.
      vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
    }
    if (depth_multiplier == 1) {
      // Write directly to the output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    } else {
      // Buffer output for subsequent reduction step.
      Eigen::internal::pstoreu<T>(out_buffer + i, vaccum);
    }
  }

  if (output_scalar_size > 0) {
    auto vaccum = Eigen::internal::pset1<Packet>(0);
    for (int j = 0; j < filter_spatial_size; ++j) {
      const int64 index =
          output_vectorized_size + j * padded_filter_inner_dim_size;
      const auto filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      const auto data_block = Eigen::internal::ploadu<Packet>(buffer + index);
      vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
    }
    // Load accumulator into an array and loop through output.
    T out_buf[kPacketSize];
    Eigen::internal::pstoreu<T>(out_buf, vaccum);
    if (depth_multiplier == 1) {
      // Write directly to the output.
      for (int j = 0; j < output_scalar_size; ++j) {
        output[base_output_index + output_vectorized_size + j] = out_buf[j];
      }
    } else {
      // Buffer output for subsequent reduction step.
      for (int j = 0; j < output_scalar_size; ++j) {
        out_buffer[output_vectorized_size + j] = out_buf[j];
      }
    }
  }

  // Iterate over 'in_depth', reduce over 'depth_multiplier', write 'output'.
  if (depth_multiplier > 1) {
    for (int64 d = 0; d < in_depth; ++d) {
      const int64 index = d * args.depth_multiplier;
      T accum = 0;
      for (int64 dm = 0; dm < dm_vectorized_size; dm += kPacketSize) {
        const auto v = Eigen::internal::ploadu<Packet>(out_buffer + index + dm);
        accum += Eigen::internal::predux(v);
      }
      // Copy scalar portion of replicated output.
      for (int64 dm = 0; dm < dm_scalar_size; ++dm) {
        accum += out_buffer[index + dm_vectorized_size + dm];
      }
      // Copy to output.
      output[base_output_index + d] = accum;
    }
  }
}

// Kernels to compute the input backprop for depthwise convolution.
template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropInputOp;

// Computes the depthwise conv2d backprop input of 'out_backprop' by
// 'depthwise_filter' and stores the result in 'in_backprop'.
template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<CPUDevice, T> {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* out_backprop, const T* depthwise_filter,
                     T* in_backprop) {
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Pad 'depthwise_filter' to vector register width (if needed).
    const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
    Tensor padded_filter;
    if (pad_filter) {
      // Allocate space for padded filter.
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &padded_filter));
      // Write out padded filter.
      functor::DepthwiseFilterPadOp<T>()(
          args, depthwise_filter, padded_filter.template flat<T>().data());
    }
    const T* filter_data =
        pad_filter ? padded_filter.template flat<T>().data() : depthwise_filter;

    // Computes one shard of depthwise conv2d backprop input.
    auto shard = [&ctx, &args, &out_backprop, &filter_data, &in_backprop](
        int64 start, int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer to copy regions from 'out_backprop'.
      Tensor out_bprop_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &out_bprop_buffer));
      T* out_bprop_buf = out_bprop_buffer.template flat<T>().data();

      // Allocate buffer for intermediate results.
      Tensor in_bprop_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({padded_filter_inner_dim_size}),
                                  &in_bprop_buffer));
      T* in_bprop_buf = in_bprop_buffer.template flat<T>().data();

      for (int64 b = start; b < limit; ++b) {
        for (int64 in_r = 0; in_r < args.in_rows; ++in_r) {
          for (int64 in_c = 0; in_c < args.in_cols; ++in_c) {
            // Populate 'out_bprop_buf' from local 'out_backprop' region.
            CopyOutputBackpropRegion<T>(
                args, padded_filter_inner_dim_size, in_r, in_c,
                out_backprop + b * output_image_size, out_bprop_buf);

            // Compute depthwise backprop input.
            ComputeBackpropInput<T>(args, padded_filter_inner_dim_size, in_r,
                                    in_c, filter_data, out_bprop_buf,
                                    in_bprop_buf,
                                    in_backprop + b * input_image_size);
          }
        }
      }
    };

    const int64 shard_cost = args.in_rows * args.in_cols * args.out_depth;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, args.batch,
          shard_cost, shard);
  }
};

template <typename T>
static void DepthwiseConvBackpropInputReference(const DepthwiseArgs& args,
                                                const T* out_backprop,
                                                const T* filter,
                                                T* in_backprop) {
  // Naive for loop as a reference point without concerns about performance.
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
                0, (in_r - args.filter_rows + args.pad_rows + stride) / stride);
            const int out_r_end =
                std::min(args.out_rows - 1, (in_r + args.pad_rows) / stride);

            for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
              const int out_c_start = std::max(
                  0,
                  (in_c - args.filter_cols + args.pad_cols + stride) / stride);
              const int out_c_end =
                  std::min(args.out_cols - 1, (in_c + args.pad_cols) / stride);

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
                        (in_d + args.in_depth * (f_c + args.filter_cols * f_r));
                sum +=
                    out_backprop[out_backprop_offset] * filter[filter_offset];
              }
            }
          }

          int in_backprop_offset =
              in_d +
              args.in_depth * (in_c + args.in_cols * (in_r + args.in_rows * b));
          in_backprop[in_backprop_offset] = sum;
        }
      }
    }
  }
}

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
    if (!context->forward_input_to_output(0, 0, &in_backprop)) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input_shape, &in_backprop));
    }
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

#define REGISTER_CPU_KERNEL(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNativeBackpropInput") \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T"),               \
                          DepthwiseConv2dNativeBackpropInputOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

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

// Computes filter backprop using 'out_backprop' and 'input_buffer', storing the
// result in 'output_buffer' at an index computed from 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   'input_buffer' [rows, cols, in_depth, depth_multiplier]
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
//   'out_backprop' [out_rows, out_cols, in_depth, depth_multiplier]
//
//     [q00, q01, q10, q11] [q20, q21, r00, r01]
//     [r10, r11, r20, r21] [s00, s01, s10, s11]
//     [s20, s21, t00, t01] [t10, t11, t20, a21]
//
//   First output register of 'filter_backprop'
//     [u0, v0, w0, x0] += ([f00, f01, f10, f11] x [q00, q01, q10, q11])
//
template <typename T>
static void ComputeBackpropFilter(const DepthwiseArgs& args,
                                  const int64 padded_out_depth_size,
                                  const int64 out_r, const int64 out_c,
                                  const T* out_backprop, const T* input_buffer,
                                  T* output_buffer) {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
  // Calculate vectorized size of 'padded_out_depth_size'.
  const int64 out_depth = args.out_depth;
  const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
  const int64 output_vectorized_size =
      (padded_out_depth_size / kPacketSize) * kPacketSize;
  const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;
  // Determine whether we can execute fast or slow code path.
  const int64 output_image_size =
      args.out_rows * args.out_cols * args.out_depth;
  const int64 output_last_vector_index =
      output_image_size - (filter_spatial_size * padded_out_depth_size);
  const bool fast_path = base_output_index <= output_last_vector_index;

  if (fast_path) {
    // TODO(andydavis) Process multiple inputs in 'input_buffer' so we can
    // amortize the cost of 'output_buffer' load store in the loop below.
    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Load vector register from 'out_backprop'.
      const auto out_bprop_block =
          Eigen::internal::ploadu<Packet>(out_backprop + base_output_index + i);
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index = i + j * padded_out_depth_size;
        // Load vector register from 'input_buffer'.
        const auto input_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Load output block into vector register.
        auto out_block_data = output_buffer + index;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        // Vector multiply-add.
        out_block = Eigen::internal::pmadd<Packet>(out_bprop_block, input_block,
                                                   out_block);
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
    }
  } else {
    // Slow path (cant do vector reads from non-padded 'out_backprop'.
    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Calculate safe read size from 'out_backprop'.
      const int64 out_bprop_index = base_output_index + i;
      const int64 out_bprop_limit =
          std::min(output_image_size, out_bprop_index + kPacketSize);
      T out_buf[kPacketSize];
      memset(&out_buf, 0, kPacketSize * sizeof(T));
      const int64 scalar_size = out_bprop_limit - out_bprop_index;
      for (int64 j = 0; j < scalar_size; ++j) {
        out_buf[j] = out_backprop[out_bprop_index + j];
      }
      // Load vector register from 'out_buf'.
      const auto out_bprop_block = Eigen::internal::ploadu<Packet>(out_buf);
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index = i + j * padded_out_depth_size;
        // Load vector register from 'input_buffer'.
        const auto input_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Load output block into vector register.
        auto out_block_data = output_buffer + index;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        // Vector multiply-add.
        out_block = Eigen::internal::pmadd<Packet>(out_bprop_block, input_block,
                                                   out_block);
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
    }
  }
}

template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropFilterOp;

template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<CPUDevice, T> {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* out_backprop, const T* input,
                     T* filter_backprop) {
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    const int64 padded_out_depth_size =
        ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

    // Allocate output buffers for each image in 'batch' (padded to vector
    // register boundaries).
    Tensor output_buffer;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({args.batch, filter_spatial_size,
                                             padded_out_depth_size}),
                                &output_buffer));
    T* output_buffer_data = output_buffer.template flat<T>().data();

    // Computes one shard of depthwise conv2d backprop filter.
    auto shard = [&ctx, &args, &out_backprop, &input, &output_buffer_data](
        int64 start, int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_out_depth_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer for local input regions.
      Tensor input_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DataTypeToEnum<T>::value,
                   TensorShape({filter_spatial_size, padded_out_depth_size}),
                   &input_buffer));
      T* input_buffer_data = input_buffer.template flat<T>().data();

      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64 padded_filter_size =
          filter_spatial_size * padded_out_depth_size;

      for (int b = start; b < limit; ++b) {
        // Initialize 'output_buffer' for 'b'.
        auto* output_buffer = output_buffer_data + b * padded_filter_size;
        memset(output_buffer, 0, padded_filter_size * sizeof(T));

        for (int out_r = 0; out_r < args.out_rows; ++out_r) {
          for (int out_c = 0; out_c < args.out_cols; ++out_c) {
            // Populate 'input_buffer_data' with data from local input region.
            functor::DepthwiseInputCopyOp<T>()(
                args, padded_out_depth_size, out_r, out_c,
                input + b * input_image_size, input_buffer_data);
            // Compute depthwise backprop filter.
            ComputeBackpropFilter(args, padded_out_depth_size, out_r, out_c,
                                  out_backprop + b * output_image_size,
                                  input_buffer_data, output_buffer);
          }
        }
      }
    };
    const int64 shard_cost = args.out_rows * args.out_cols * args.out_depth;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, args.batch,
          shard_cost, shard);

    // Accumulate 'output_buffer' from each shard into 'output'.
    const int64 out_depth = args.out_depth;
    const int64 vectorized_size = (out_depth / kPacketSize) * kPacketSize;
    const int64 scalar_size = out_depth - vectorized_size;
    const int64 padded_filter_size =
        filter_spatial_size * padded_out_depth_size;
    memset(filter_backprop, 0, filter_spatial_size * out_depth * sizeof(T));

    for (int64 i = 0; i < filter_spatial_size; ++i) {
      const int64 buffer_base = i * padded_out_depth_size;
      const int64 output_base = i * out_depth;
      // Write vectorized length of filter's inner dimension to output.
      for (int64 j = 0; j < vectorized_size; j += kPacketSize) {
        // Load data from 'filter_backprop' into vector register.
        auto out_block_data = filter_backprop + output_base + j;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        for (int b = 0; b < args.batch; ++b) {
          // Load data from 'output_buffer' for 'b'.
          const auto* output_buffer =
              output_buffer_data + b * padded_filter_size;
          const auto v =
              Eigen::internal::ploadu<Packet>(output_buffer + buffer_base + j);
          // Add 'v' to 'out_block'.
          out_block = Eigen::internal::padd<Packet>(out_block, v);
        }
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64 j = 0; j < scalar_size; ++j) {
        for (int b = 0; b < args.batch; ++b) {
          const auto* output_buffer =
              output_buffer_data + b * padded_filter_size;
          filter_backprop[output_base + vectorized_size + j] +=
              output_buffer[buffer_base + vectorized_size + j];
        }
      }
    }
  }
};

template <typename T>
static void DepthwiseConvBackpropFilterReference(const DepthwiseArgs& args,
                                                 const T* out_backprop,
                                                 const T* input,
                                                 T* filter_backprop) {
  int num_filter_backprop = args.filter_rows * args.filter_cols *
                            args.in_depth * args.depth_multiplier;
  memset(filter_backprop, 0, num_filter_backprop * sizeof(T));
  // Naive for loop as a reference point without concerns about performance.
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
                        (in_d + args.in_depth * (f_c + args.filter_cols * f_r));
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

// Kernel to compute the filter backprop for depthwise convolution.
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
    if (!context->forward_input_to_output(1, 0, &filter_backprop)) {
      OP_REQUIRES_OK(
          context, context->allocate_output(0, filter_shape, &filter_backprop));
    }

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

#define REGISTER_CPU_KERNEL(T)                    \
  REGISTER_KERNEL_BUILDER(                        \
      Name("DepthwiseConv2dNativeBackpropFilter") \
          .Device(DEVICE_CPU)                     \
          .TypeConstraint<T>("T"),                \
      DepthwiseConv2dNativeBackpropFilterOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

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
