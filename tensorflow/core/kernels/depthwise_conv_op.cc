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
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchDepthwiseConvOp;

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
  static void Run(const DepthwiseArgs& args, const T* filter,
                  T* padded_filter) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar lengths of filter's inner dimension.
    const int64 filter_inner_dim_size = args.out_depth;
    const int64 vectorized_size =
        (filter_inner_dim_size / kPacketSize) * kPacketSize;
    const int64 scalar_size = filter_inner_dim_size - vectorized_size;
    // Calculate required padding and padded output buffer stride.
    const int64 pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
    const int64 padded_filter_stride = vectorized_size + kPacketSize;

    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    for (int64 i = 0; i < filter_spatial_size; ++i) {
      const int64 input_base = i * filter_inner_dim_size;
      const int64 output_base = i * padded_filter_stride;
      // Write vectorized length of filter's inner dimension to output.
      for (int64 j = 0; j < vectorized_size; j += kPacketSize) {
        const auto v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
        Eigen::internal::pstoreu<T>(padded_filter + output_base + j, v);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64 j = 0; j < scalar_size; ++j) {
        padded_filter[output_base + vectorized_size + j] =
            filter[input_base + vectorized_size + j];
      }
      // Pad the remainder of output to vector-register boundary.
      for (int64 j = 0; j < pad_size; ++j) {
        padded_filter[output_base + vectorized_size + scalar_size + j] = 0;
      }
    }
  }
};

// Copies data from local region in 'input' specified by 'out_r' and 'out_'c'
// to 'input_buffer'. The copied data is replicated by factor
// 'args.depth_mulitplier', and padded to vector register-width boundaries so
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
struct InputBufferCopyOp {
  static void Run(const DepthwiseArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* input, T* input_buffer) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64 input_vectorized_size =
        (args.in_depth / kPacketSize) * kPacketSize;
    const int64 input_scalar_size = args.in_depth % kPacketSize;

    // Calculate vectorized and scalar (residual) lengths for
    // 'depth_multiplier'. This is used to efficiently replicate data for
    // when 'depth_multiplier' > kPacketSize.
    const int64 dm_vectorized_size =
        (args.depth_multiplier / kPacketSize) * kPacketSize;
    const int64 dm_scalar_size = args.depth_multiplier % kPacketSize;

    // Calculate output padding length.
    const int64 output_scalar_size = args.out_depth % kPacketSize;
    const int64 output_pad_size =
        output_scalar_size > 0 ? kPacketSize - output_scalar_size : 0;

    const int64 replicated_packet_size = kPacketSize * args.depth_multiplier;

    // Iterate through all rows x cols reading 'in_depth' from 'input' and
    // replicating by 'depth_multiplier' into 'input_buffer' (otherwise
    // zero-padding input buffer as needed).
    auto* in_buf = input_buffer;
    const int64 in_r_start = out_r * args.stride - args.pad_rows;
    const int64 in_c_start = out_c * args.stride - args.pad_cols;

    for (int64 f_r = 0; f_r < args.filter_rows; ++f_r) {
      const int64 in_r = in_r_start + f_r;

      for (int64 f_c = 0; f_c < args.filter_cols; ++f_c) {
        const int64 in_c = in_c_start + f_c;

        if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
            in_c < args.in_cols) {
          auto* in = input + (in_r * args.in_cols + in_c) * args.in_depth;
          // Copy vectorized portion of inner dimension.
          for (int64 d = 0; d < input_vectorized_size; d += kPacketSize) {
            auto v = Eigen::internal::ploadu<Packet>(in + d);
            for (int dm = 0; dm < args.depth_multiplier; ++dm) {
              Eigen::internal::pscatter<T, Packet>(in_buf + dm, v,
                                                   args.depth_multiplier);
            }
            in_buf += replicated_packet_size;
          }

          // Copy scalar portion of inner dimension.
          for (int64 d = 0; d < input_scalar_size; ++d) {
            T v = in[input_vectorized_size + d];
            const int64 base = d * args.depth_multiplier;
            if (dm_vectorized_size > 0) {
              // Copy vectorized portion of replicated output.
              // This branch is only taken if 'args.depth_multiplier' is
              // vectorizable (i.e. args.depth_multiplier >= register width).
              auto p = Eigen::internal::pset1<Packet>(v);
              for (int64 dm = 0; dm < dm_vectorized_size; dm += kPacketSize) {
                Eigen::internal::pstoreu<T>(in_buf + base + dm, p);
              }
              // Copy scalar portion of replicated output.
              for (int64 dm = 0; dm < dm_scalar_size; ++dm) {
                in_buf[base + dm_vectorized_size + dm] = v;
              }
            } else {
              // Depth multiplier is less than one packet: scalar copy.
              for (int dm = 0; dm < args.depth_multiplier; ++dm) {
                in_buf[base + dm] = v;
              }
            }
          }
          in_buf += input_scalar_size * args.depth_multiplier;

          // Pad the remainder of the output to vector register boundary.
          for (int64 d = 0; d < output_pad_size; ++d) {
            in_buf[d] = 0;
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
};

// Computes the vectorized product of 'input_buffer' and 'filter' and stores
// result in 'output' at location specified by 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   input_buffer [rows, cols, in_depth, depth_multiplier]
//     [a0, a0, a1, a1] [a2, a2, 0, 0] [b0, b0, b1, b1] [b2, b2, 0, 0]
//     [e0, e0, e1, e1] [e2, e2, 0, 0] [f0, f0, f1, f1] [f2, f2, 0, 0]
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q0, q1, q2, q3] = ([a0, a0, a1, a1] x [u0, v0, w0, x0]) +
//                        ([b0, b0, b1, b1] x [u1, v1, w1, x1]) +
//                        ([e0, e0, e1, e1] x [u2, v2, w2, x2]) +
//                        ([f0, f0, f1, f1] x [u3, v3, w3, x3])
//
// TODO(andydavis) Experiment with processing multiple inputs per input buffer.
template <typename T>
struct DepthwiseConv2DKernel {
  static void Run(const DepthwiseArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* filter, const T* input_buffer,
                  T* output) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 out_depth = args.out_depth;
    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    const int64 output_scalar_size = out_depth % kPacketSize;
    const int64 output_vectorized_size =
        (out_depth / kPacketSize) * kPacketSize;
    const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;

    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Reset accumulator.
      auto vaccum = Eigen::internal::pset1<Packet>(0);
      for (int j = 0; j < filter_spatial_size; ++j) {
        // Calculate index.
        const int64 index = i + j * padded_filter_inner_dim_size;
        // Load filter.
        // TODO(andydavis) Unroll 'out_c' loop in caller so we can load
        // multiple inputs here to amortize the cost of each filter block load.
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        // Load input.
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Vector multiply-add.
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Store vector accumulator to output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    }

    if (output_scalar_size > 0) {
      auto vaccum = Eigen::internal::pset1<Packet>(0);
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index =
            output_vectorized_size + j * padded_filter_inner_dim_size;
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Load accumulator into an array and loop through output.
      T out_buf[kPacketSize];
      Eigen::internal::pstoreu<T>(out_buf, vaccum);
      const int64 last_output_index =
          base_output_index + output_vectorized_size;
      for (int j = 0; j < output_scalar_size; ++j) {
        output[last_output_index + j] = out_buf[j];
      }
    }
  }
};

// Computes the depthwise conv2d of 'input' by 'depthwise_filter' and stores
// the result in 'output'. This implementation trades off copying small patches
// of the input to achieve better data alignment, which enables vectorized
// load/store and multiply-add operations (see comments at InputBufferCopyOp and
// DepthwiseConv2DKernel for details).
//
// TODO(andydavis) Evaluate the performance of processing multiple input
// patches in the inner loop.
// TODO(andydavis) Consider a zero-copy implementation for the case when
// 'in_depth' is a multiple of register width, and 'depth_multipler' is one.
// TODO(andydavis) Evaluate the performance of alternative implementations.
template <typename T>
struct LaunchDepthwiseConvOp<CPUDevice, T> {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* input, const T* depthwise_filter, T* output) {
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
      DepthwiseFilterPadOp<T>::Run(args, depthwise_filter,
                                   padded_filter.template flat<T>().data());
    }
    const T* filter_data =
        pad_filter ? padded_filter.template flat<T>().data() : depthwise_filter;

    // Computes one shard of depthwise conv2d output.
    auto shard = [&ctx, &args, &input, &filter_data, &output](int64 start,
                                                              int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer for local input regions.
      Tensor input_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &input_buffer));
      T* input_buffer_data = input_buffer.template flat<T>().data();

      for (int64 b = start; b < limit; ++b) {
        const int64 in_base = b * input_image_size;
        const int64 out_base = b * output_image_size;

        for (int64 out_r = 0; out_r < args.out_rows; ++out_r) {
          for (int64 out_c = 0; out_c < args.out_cols; ++out_c) {
            // Populate 'input_buffer_data' with data from local input region.
            InputBufferCopyOp<T>::Run(args, padded_filter_inner_dim_size, out_r,
                                      out_c, input + in_base,
                                      input_buffer_data);

            // Process buffered input across all filters and store to output.
            DepthwiseConv2DKernel<T>::Run(args, padded_filter_inner_dim_size,
                                          out_r, out_c, filter_data,
                                          input_buffer_data, output + out_base);
          }
        }
      }
    };

    // TODO(andydavis) Shard over batch X out_rows (instead of just batch).
    const int64 total_shards = args.batch;
    const int64 shard_cost = args.out_rows * args.out_cols * args.out_depth;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

#if GOOGLE_CUDA

template <typename T>
struct DepthwiseConv2dGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args, const T* input,
                  const T* filter, T* output);
};

template <typename T>
struct LaunchDepthwiseConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const T* input, const T* filter, T* output) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    DepthwiseConv2dGPULaunch<T>().Run(d, args, input, filter, output);
    auto stream = ctx->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(
        ctx, stream->ok(),
        errors::Internal(
            "Launch of gpu kernel for DepthwiseConv2dGPULaunch failed"));
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
    auto stream = ctx->op_device_context<GPUDeviceContext>()->stream();
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
    auto stream = ctx->op_device_context<GPUDeviceContext>()->stream();

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
