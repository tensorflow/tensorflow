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
      functor::DepthwiseFilterPadOp<T>()(
          args, depthwise_filter, padded_filter.template flat<T>().data());
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
    auto stream = ctx->op_device_context()->stream();
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

}  // namespace tensorflow
