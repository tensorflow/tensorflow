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

#include "tensorflow/core/kernels/depthwise_conv_op.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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
                  const int64_t padded_filter_inner_dim_size,
                  const int64_t out_r, const int64_t out_c, const T* filter,
                  const T* input_buffer, T* output, TensorFormat data_format) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64_t out_depth = args.out_depth;
    const int64_t filter_spatial_size = args.filter_rows * args.filter_cols;
    const int64_t output_scalar_size = out_depth % kPacketSize;
    const int64_t output_vectorized_size =
        (out_depth / kPacketSize) * kPacketSize;
    const int64_t base_output_index =
        (out_r * args.out_cols + out_c) * out_depth;

    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Reset accumulator.
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        // Calculate index.
        const int64_t index = i + j * padded_filter_inner_dim_size;
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
      auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64_t index =
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
      const int64_t last_output_index =
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

  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* input, const T* depthwise_filter, T* output,
                  TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on CPU is only supported for NHWC format"));
    static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    // Pad 'depthwise_filter' to vector register width (if needed).
    const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
    Tensor padded_filter;
    if (pad_filter) {
      // Allocate space for padded filter.
      const int64_t filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64_t padded_filter_inner_dim_size =
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
    auto shard = [&ctx, &args, &input, &filter_data, &output, data_format](
                     int64_t start, int64_t limit) {
      static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64_t input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64_t output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64_t filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64_t padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer for local input regions.
      Tensor input_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &input_buffer));
      T* input_buffer_data = input_buffer.template flat<T>().data();

      for (int64_t i = start; i < limit; ++i) {
        const int64_t b = i / args.out_rows;
        const int64_t in_base = b * input_image_size;
        const int64_t out_base = b * output_image_size;

        const int64_t out_r = i % args.out_rows;

        for (int64_t out_c = 0; out_c < args.out_cols; ++out_c) {
          // Populate 'input_buffer_data' with data from local input region.
          functor::DepthwiseInputCopyOp<T>()(args, padded_filter_inner_dim_size,
                                             out_r, out_c, input + in_base,
                                             input_buffer_data);

          // Process buffered input across all filters and store to output.
          DepthwiseConv2DKernel<T>::Run(
              args, padded_filter_inner_dim_size, out_r, out_c, filter_data,
              input_buffer_data, output + out_base, data_format);
        }
      }
    };

    const int64_t total_shards = args.batch * args.out_rows;

    // Empirically tested to give reasonable performance boosts at batch size 1
    // without reducing throughput at batch size 32.
    const float kCostMultiplier = 2.5f;

    // TODO(andydavis): Estimate shard cost (in cycles) based on the number of
    // flops/loads/stores required to compute one shard.
    const int64_t shard_cost = kCostMultiplier * args.out_cols * args.out_depth;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

// Extern template instantiated in conv_ops.cc.
extern template struct LaunchConv2DOp<CPUDevice, Eigen::half>;
extern template struct LaunchConv2DOp<CPUDevice, float>;
extern template struct LaunchConv2DOp<CPUDevice, double>;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Extern template instantiated in conv_ops.cc.
extern template struct LaunchConv2DOp<GPUDevice, Eigen::half>;
extern template struct LaunchConv2DOp<GPUDevice, float>;
extern template struct LaunchConv2DOp<GPUDevice, double>;

// Extern template instantiated in depthwise_conv_op_gpu.cc.
extern template struct LaunchDepthwiseConvOp<GPUDevice, Eigen::half>;
extern template struct LaunchDepthwiseConvOp<GPUDevice, float>;
extern template struct LaunchDepthwiseConvOp<GPUDevice, double>;

bool DisableDepthwiseConvDeterminismExceptions() {
  static bool cached_disable = [] {
    bool disable = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_DISABLE_DEPTHWISE_CONV_DETERMINISM_EXCEPTIONS",
        /*default_val*/ false, &disable));
    return disable;
  }();
  return cached_disable;
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class DepthwiseConv2dNativeOp : public BinaryOp<T> {
 public:
  explicit DepthwiseConv2dNativeOp(OpKernelConstruction* context)
      : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    stride_ = GetTensorDim(strides_, data_format_, 'H');
    const int64_t stride_w = GetTensorDim(strides_, data_format_, 'W');
    const int64_t stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64_t stride_c = GetTensorDim(strides_, data_format_, 'C');

    OP_REQUIRES(context, stride_ == stride_w,
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));

    cudnn_use_autotune_ = CudnnUseAutotune();
    dtype_ = DataTypeToEnum<T>::value;
#if CUDNN_VERSION >= 8000
    // From the cuDNN release note 8.0: We’ve extended the fprop and dgrad
    // NHWC depthwise kernels to support more combinations (filter
    // sizes/strides) such as 5x5/1x1, 5x5/2x2, 7x7/1x1, 7x7/2x2 (in addition
    // to what we already have, 1x1/1x1, 3x3/1x1, 3x3/2x2), which provides
    // good performance. (https://docs.nvidia.com/deeplearning/sdk/cudnn-
    // release-notes/rel_8.html#rel_8)
    use_cudnn_grouped_conv_ =
        dtype_ == DT_HALF &&
        (data_format_ == FORMAT_NCHW ||
         (data_format_ == FORMAT_NHWC && stride_ == stride_w &&
          (stride_ == 1 || stride_ == 2)));
#elif CUDNN_VERSION >= 7603
    // Use CuDNN grouped conv only when input/output is NCHW and float16(half).
    // See cudnn release note 7.6.3. (https://docs.nvidia.com/deeplearning/sdk/c
    // udnn-release-notes/rel_763.html#rel_763)
    use_cudnn_grouped_conv_ = dtype_ == DT_HALF && data_format_ == FORMAT_NCHW;
#else
    use_cudnn_grouped_conv_ = false;
#endif
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // in_depth for input and filter must match.
    const int64_t in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int32_t depth_multiplier = filter.dim_size(3);

    // The output depth is input depth x depth multiplier
    const int32_t out_depth = in_depth * depth_multiplier;

    const int64_t input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int32_t input_rows = static_cast<int32>(input_rows_raw);
    const int32_t filter_rows = filter.dim_size(0);

    const int64_t input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int32_t input_cols = static_cast<int32>(input_cols_raw);
    const int32_t filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int32_t batch = input.dim_size(0);

    int64_t out_rows = 0, out_cols = 0, pad_top = 0, pad_bottom = 0,
            pad_left = 0, pad_right = 0;
    if (padding_ == Padding::EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'H', &pad_top,
                               &pad_bottom);
      GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'W', &pad_left,
                               &pad_right);
    }
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                input_rows, filter_rows, stride_, padding_,
                                &out_rows, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                input_cols, filter_cols, stride_, padding_,
                                &out_cols, &pad_left, &pad_right));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
    OP_REQUIRES(
        context,
        (!std::is_same<Device, GPUDevice>::value ||
         FastBoundsCheck(out_shape.num_elements(),
                         std::numeric_limits<int32>::max())),
        errors::InvalidArgument("Output elements too large for GPU kernel"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // TODO(csigg): Have autotune decide if native is faster than cuDNN.
    // If in_depth==1, this operation is just a standard convolution.
    // Depthwise convolution is a special case of cuDNN's grouped convolution.
    bool use_cudnn = std::is_same<Device, GPUDevice>::value &&
                     (in_depth == 1 ||
                      (use_cudnn_grouped_conv_ &&
                       IsCudnnSupportedFilterSize(/*filter_rows=*/filter_rows,
                                                  /*filter_cols=*/filter_cols,
                                                  /*in_depth=*/in_depth,
                                                  /*out_depth=*/out_depth)));

    VLOG(2) << "DepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; Output: [" << batch << ", " << out_rows << ", " << out_cols
            << ", " << out_depth << "], stride = " << stride_
            << ", pad_top = " << pad_top << ", pad_left = " << pad_left
            << ", Use cuDNN: " << use_cudnn;

    if (use_cudnn) {
      // Reshape from TF depthwise filter to cuDNN grouped convolution filter:
      //
      //                  | TensorFlow       | cuDNN
      // --------------------------------------------------------------------
      // filter_out_depth | depth_multiplier | depth_multiplier * group_count
      // filter_in_depth  | in_depth         | in_depth / group_count
      //
      // For depthwise convolution, we have group_count == in_depth.
      int32_t filter_in_depth = 1;
      TensorShape shape =
          TensorShape{filter_rows, filter_cols, filter_in_depth, out_depth};
      Tensor reshaped_filter(/*type=*/dtype_);
      OP_REQUIRES(
          context, reshaped_filter.CopyFrom(filter, shape),
          errors::Internal(
              "Failed to reshape filter tensor for grouped convolution."));
      // TODO(yangzihao): Send in arbitrary dilation rates after the dilated
      // conv is supported.
      launcher_(context, /*use_cudnn=*/true, cudnn_use_autotune_, input,
                reshaped_filter, /*row_dilation=*/1, /*col_dilation=*/1,
                stride_, stride_, padding_, explicit_paddings_, output,
                data_format_);
      return;
    }

    DepthwiseArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = stride_;
    args.pad_rows = pad_top;
    args.pad_cols = pad_left;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    auto input_ptr = input.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();
    LaunchDepthwiseConvOp<Device, T>()(context, args, input_ptr, filter_ptr,
                                       output_ptr, data_format_);
  }

 protected:
  bool use_cudnn_grouped_conv_;

 private:
  std::vector<int32> strides_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;

  int64_t stride_;  // in height/width dimension.

  // For in_depth == 1 and grouped convolutions.
  LaunchConv2DOp<Device, T> launcher_;
  bool cudnn_use_autotune_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeOp);
};

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DepthwiseConv2dNativeOp<CPUDevice, T>)

TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
#if !defined(PLATFORM_WINDOWS) || !defined(_DEBUG)
TF_CALL_double(REGISTER_CPU_KERNEL);
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DepthwiseConv2dNativeOp<GPUDevice, T>)

TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);

#if CUDNN_VERSION >= 7000
template <typename T>
class DepthwiseConv2dGroupedConvOp
    : public DepthwiseConv2dNativeOp<GPUDevice, T> {
 public:
  DepthwiseConv2dGroupedConvOp(OpKernelConstruction* context)
      : DepthwiseConv2dNativeOp<GPUDevice, T>(context) {
    this->use_cudnn_grouped_conv_ = true;
  }
};

#define REGISTER_GROUPED_CONV_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")            \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .Label("cudnn_grouped_convolution"), \
                          DepthwiseConv2dGroupedConvOp<T>)

TF_CALL_half(REGISTER_GROUPED_CONV_KERNEL);
TF_CALL_float(REGISTER_GROUPED_CONV_KERNEL);
TF_CALL_double(REGISTER_GROUPED_CONV_KERNEL);
#endif  // CUDNN_VERSION
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
