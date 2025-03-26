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

#ifndef TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
#define TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int stride;
  int pad_rows;  // Amount of padding to the top of the input
  int pad_cols;  // Amount of padding to the left of the input

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  DepthwiseArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        depth_multiplier(0),
        stride(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchDepthwiseConvOp {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* input, const T* filter, T* output,
                  TensorFormat data_format);
};

template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropInputOp {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* filter, T* in_backprop,
                  TensorFormat data_format);
};

template <typename Device, typename T>
struct LaunchDepthwiseConvBackpropFilterOp {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* input, T* filter_backprop,
                  TensorFormat data_format);
};

bool UseCudnnWith16BitFloat(OpKernelContext* ctx, DataType dtype);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
struct LaunchDepthwiseConvOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* input, const T* filter, T* output,
                  TensorFormat data_format);
};

template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<Eigen::GpuDevice, T> {
  void operator()(class OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* filter, T* in_backprop,
                  TensorFormat data_format);
};

template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<Eigen::GpuDevice, T> {
  void operator()(class OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* input, T* filter_backprop,
                  TensorFormat data_format);
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

namespace tensorflow {
namespace functor {

// Pads 'filter' to vector-register boundary along its inner dimension:
//   filter_inner_dim_size = in_depth * depth_multiplier
// Requires 'filter' to have the following storage order:
//   [filter_rows, filter_cols, in_depth, depth_multiplier]
// Returns zero-padded filter in 'padded_filter'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   So we have a total of 3 * 2 = 6 filters, each of spatial size 2 x 2.
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, u1, v1] [w1, x1, y1, z1]
//     [u2, v2, w2, x2] [y2, z2, u3, v3] [w3, x3, y3, z3]
//
//   padded_filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]

template <typename T>
struct DepthwiseFilterPadOp {
  void operator()(const DepthwiseArgs& args, const T* filter,
                  T* padded_filter) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar lengths of filter's inner dimension.
    const int64_t filter_inner_dim_size = args.out_depth;
    const int64_t vectorized_size =
        (filter_inner_dim_size / kPacketSize) * kPacketSize;
    const int64_t scalar_size = filter_inner_dim_size - vectorized_size;
    // Calculate required padding and padded output buffer stride.
    const int64_t pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
    const int64_t padded_filter_stride = vectorized_size + kPacketSize;

    const int64_t filter_spatial_size = args.filter_rows * args.filter_cols;
    for (int64_t i = 0; i < filter_spatial_size; ++i) {
      const int64_t input_base = i * filter_inner_dim_size;
      const int64_t output_base = i * padded_filter_stride;
      // Write vectorized length of filter's inner dimension to output.
      for (int64_t j = 0; j < vectorized_size; j += kPacketSize) {
        const auto v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
        Eigen::internal::pstoreu<T>(padded_filter + output_base + j, v);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64_t j = 0; j < scalar_size; ++j) {
        padded_filter[output_base + vectorized_size + j] =
            filter[input_base + vectorized_size + j];
      }
      // Pad the remainder of output to vector-register boundary.
      for (int64_t j = 0; j < pad_size; ++j) {
        padded_filter[output_base + vectorized_size + scalar_size + j] =
            static_cast<T>(0);
      }
    }
  }
};

// Copies data from local region in 'input' specified by 'out_r' and 'out_'c'
// to 'input_buffer'. The copied data is replicated by factor
// 'args.depth_multiplier', and padded to vector register-width boundaries so
// that it is aligned for efficient traversal and vector multiply-add by the
// depthwise kernel.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//
//   input: [batch, in_rows, in_cols, in_depth]
//
//     [a0, a1, a2, b0, b1, b2, ..., e0, e1, e2, f0, f1, f2, ...]
//
//   input_buffer (register boundaries shown):
//     [a0, a0, a1, a1] [a2, a2, 0, 0]   in_row = 0, in_col = 0
//     [b0, b0, b1, b1] [b2, b2, 0, 0]   in_row = 0, in_col = 1
//     [e0, e0, e1, e1] [e2, e2, 0, 0]   in_row = 1, in_col = 0
//     [f0, f0, f1, f1] [f2, f2, 0, 0]   in_row = 1, in_col = 1
//
// Returns replicated and padded data from specified input region in
// 'input_buffer'.

template <typename T>
struct DepthwiseInputCopyOp {
  void operator()(const DepthwiseArgs& args,
                  const int64_t padded_filter_inner_dim_size,
                  const int64_t out_r, const int64_t out_c, const T* input,
                  T* input_buffer) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64_t kPacketSize = Eigen::internal::packet_traits<T>::size;

    const int64_t kDepth = args.depth_multiplier;
    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64_t input_vectorized_size =
        (args.in_depth / kPacketSize) * kPacketSize;
    const int64_t input_scalar_size = args.in_depth - input_vectorized_size;

    // Calculate output padding length.
    const int64_t output_scalar_size = args.out_depth % kPacketSize;
    const int64_t output_pad_size =
        output_scalar_size > 0 ? kPacketSize - output_scalar_size : 0;

    // Iterate through all rows x cols reading 'in_depth' from 'input' and
    // replicating by 'depth_multiplier' into 'input_buffer' (otherwise
    // zero-padding input buffer as needed).
    auto* in_buf = input_buffer;
    const int64_t in_r_start = out_r * args.stride - args.pad_rows;
    const int64_t in_c_start = out_c * args.stride - args.pad_cols;

    // TODO: add a ploaddup variant for depth == 2 if needed.
    if (kDepth > 1 && kDepth <= kPacketSize) {
      for (int64_t f_r = 0; f_r < args.filter_rows; ++f_r) {
        const int64_t in_r = in_r_start + f_r;
        const auto* in = input + in_r * args.in_cols * args.in_depth +
                         in_c_start * args.in_depth;
        const int64_t step = args.in_depth;

        for (int64_t f_c = 0; f_c < args.filter_cols; ++f_c, in += step) {
          // 'in' is input + (in_r * args.in_cols + in_c) * args.in_depth
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
              in_c < args.in_cols) {
            int64_t limit = args.in_depth;
            // This will overwrite up to kPacketSize next elements,
            // this is ok on all iterations except the last one, since
            // we will write correct values on a next iteration.
            if (f_c == args.filter_cols - 1) {
              limit -= (kPacketSize - kDepth) / kDepth + 1;
              if (limit < 0) {
                limit = 0;
              }
            }
            // Copy vectorized portion of inner dimension.
            for (int64_t d = 0; d < limit; d++) {
              const auto p = Eigen::internal::pset1<Packet>(in[d]);
              Eigen::internal::pstoreu<T>(in_buf, p);
              in_buf += kDepth;
            }

            // Copy the scalar portion.
            for (int64_t d = limit; d < args.in_depth; d++) {
              const auto value = in[d];
              for (int64_t dm = 0; dm < kDepth; dm++) {
                in_buf[dm] = value;
              }
              in_buf += kDepth;
            }

            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d) {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          } else {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    } else if (kDepth > kPacketSize) {
      // Calculate vectorized and scalar (residual) lengths for
      // 'depth_multiplier'. This is used to efficiently replicate data for
      // when 'depth_multiplier' > kPacketSize.
      const int64_t dm_vectorized_size = (kDepth / kPacketSize) * kPacketSize;

      const int64_t col_step = args.in_cols * args.in_depth;
      const int64_t step = args.in_depth;
      const auto start = in_c_start * step;
      const auto* in_col = input + in_r_start * col_step;
      for (int64_t f_r = 0; f_r < args.filter_rows; ++f_r, in_col += col_step) {
        const int64_t in_r = in_r_start + f_r;
        const auto* in = in_col + start;

        for (int64_t f_c = 0; f_c < args.filter_cols; ++f_c, in += step) {
          // 'in' is input + (in_r * args.in_cols + in_c) * args.in_depth
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
              in_c < args.in_cols) {
            // Copy vectorized portion of inner dimension.
            for (int64_t d = 0; d < args.in_depth; d++) {
              const auto p = Eigen::internal::pset1<Packet>(in[d]);
              for (int64_t dm = 0; dm < dm_vectorized_size; dm += kPacketSize) {
                Eigen::internal::pstoreu<T>(in_buf + dm, p);
              }
              // Overlapping store for the remainder.
              Eigen::internal::pstoreu<T>(in_buf + kDepth - kPacketSize, p);
              in_buf += kDepth;
            }
            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d) {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          } else {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    } else if (kDepth == 1) {
      const int64_t col_step = args.in_cols * args.in_depth;
      const int64_t step = args.in_depth;
      const auto start = in_c_start * step;
      const auto* in_col = input + in_r_start * col_step;
      for (int64_t f_r = 0; f_r < args.filter_rows; ++f_r, in_col += col_step) {
        const int64_t in_r = in_r_start + f_r;
        const auto* in = in_col + start;
        for (int64_t f_c = 0; f_c < args.filter_cols; ++f_c, in += step) {
          // 'in' is input + (in_r * args.in_cols + in_c) * args.in_depth
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
              in_c < args.in_cols) {
            for (int64_t d = 0; d < input_vectorized_size; d += kPacketSize) {
              const auto p = Eigen::internal::ploadu<Packet>(in + d);
              Eigen::internal::pstoreu<T>(in_buf, p);
              in_buf += kPacketSize;
            }
            for (int64_t d = 0; d < input_scalar_size; ++d) {
              T v = in[input_vectorized_size + d];
              in_buf[d] = v;
            }
            in_buf += input_scalar_size;

            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d) {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          } else {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_H_
