/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implements convolution operations with other kernels baked into the
// processing, to optimize latency and memory usage:
//  - Conv2D + BiasAdd + <Activation>
//  - Conv2D + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...
//
// Kernels for convolutions fused with image transformations (resize and mirror
// padding) defined in `conv_ops_fused_image_transform.cc`.
//
// For the CPU device we implement fusion with an Eigen tensor contraction
// output kernel. For the GPU device we rely on CuDNN primitives.
//
// NOTE: GPU only supports fusion of Conv2D + BiasAdd + <optional Relu>.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_autotune_maps.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchFusedConv2DOp {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output);
};

// This is CPU-only implementation that uses Eigen contraction output kernels.
//
// Dispatch 2D convolution to the appropriate primitive operation:
//   (1) MatMul for the case of 1x1 convolution.
//   (2) MatMul for the case when filter size equals to the input size.
//   (3) General spatial 2D convolution for all other cases.
template <typename T>
class LaunchFusedConv2DWithOutputKernel {
 public:
  LaunchFusedConv2DWithOutputKernel(
      int row_stride, int col_stride,      //
      int row_dilation, int col_dilation,  //
      Padding padding, const std::vector<int64_t>& explicit_paddings)
      : row_stride_(row_stride),
        col_stride_(col_stride),
        row_dilation_(row_dilation),
        col_dilation_(col_dilation),
        padding_(padding),
        explicit_paddings_(explicit_paddings) {}

  template <typename OutputKernel>
  void operator()(const OutputKernel& output_kernel, OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter, Tensor* output) {
    // Wrap output_kernel into type erased wrapper to reduce the number of
    // unique template instantiations for Eigen Tensor contraction expressions.
    OutputKernelWrapper output_kernel_wrapper(
        [&output_kernel](
            const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
            const Eigen::TensorContractionParams& params, Eigen::Index i,
            Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) {
          output_kernel(output_mapper, params, i, j, num_rows, num_cols);
        });

    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        row_stride_ == 1 && col_stride_ == 1 && padding_ != EXPLICIT) {
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernelWrapper>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair, std::move(output_kernel_wrapper));

    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation_ == 1 &&
               col_dilation_ == 1 && padding_ == VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernelWrapper>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair,
          std::move(output_kernel_wrapper));

    } else {
      if (padding_ == EXPLICIT) {
        functor::SpatialConvolution<CPUDevice, T, OutputKernelWrapper>()(
            ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_,
            col_stride_, row_dilation_, col_dilation_,
            static_cast<int>(explicit_paddings_[2]),
            static_cast<int>(explicit_paddings_[3]),
            static_cast<int>(explicit_paddings_[4]),
            static_cast<int>(explicit_paddings_[5]),
            std::move(output_kernel_wrapper));
      } else {
        functor::SpatialConvolution<CPUDevice, T, OutputKernelWrapper>()(
            ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_,
            col_stride_, row_dilation_, col_dilation_,
            BrainPadding2EigenPadding(padding_),
            std::move(output_kernel_wrapper));
      }
    }
  }

 private:
  // Wrap output_kernel into type erased struct to reduce the number of unique
  // template instantiations for Eigen Tensor contraction expressions.
  //
  // We do not pass std::function directly as an output kernel because it blows
  // up the binary size in debug mode with super long symbol names.
  struct OutputKernelWrapper {
    using OutputKernelFn =
        std::function<void(const ContractionOutputMapper<T, Eigen::Index>&,
                           const Eigen::TensorContractionParams&, Eigen::Index,
                           Eigen::Index, Eigen::Index, Eigen::Index)>;

    explicit OutputKernelWrapper(OutputKernelFn fn)
        : output_kernel_fn(std::move(fn)) {}

    void operator()(
        const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
        const Eigen::TensorContractionParams& params, Eigen::Index i,
        Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) const {
      output_kernel_fn(output_mapper, params, i, j, num_rows, num_cols);
    }

    OutputKernelFn output_kernel_fn;
  };

  int row_stride_;
  int col_stride_;
  int row_dilation_;
  int col_dilation_;
  const Padding padding_;
  const std::vector<int64_t>& explicit_paddings_;
};

template <typename T>
struct LaunchFusedConv2DOp<CPUDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, const FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output) {
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(context, params.data_format == FORMAT_NHWC,
                errors::Unimplemented("Fused conv implementation only supports "
                                      "NHWC tensor format for now."));
    OP_REQUIRES(context, DataTypeToEnum<T>::value != DT_HALF,
                errors::Unimplemented("Fused conv implementation with half "
                                      "precision is not supported on CPU."));

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      if (fusion == FusedComputationType::kBiasAddWithLeakyRelu) {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args,
                                                &fusion_args.leakyrelu_alpha));
      } else {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
      }
    }

    FusedBatchNormArgs<T> fused_batch_norm_args;
    if (FusedBatchNormArgs<T>::IsSupported(fusion)) {
      if (fusion == FusedComputationType::kFusedBatchNormWithLeakyRelu) {
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                              &fused_batch_norm_args,
                                              &fusion_args.leakyrelu_alpha));
      } else {
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                              &fused_batch_norm_args));
      }
    }

    LaunchFusedConv2DWithOutputKernel<T> conv2d(
        dimensions.stride_rows, dimensions.stride_cols,
        dimensions.dilation_rows, dimensions.dilation_cols, params.padding,
        params.explicit_paddings);

    switch (fusion) {
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      case FusedComputationType::kBiasAdd:
        conv2d(WithBiasAdd<T>(bias_add_args), context, input, filter, output);
        break;
      case FusedComputationType::kBiasAddWithRelu:
        conv2d(WithBiasAddAndRelu<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        conv2d(WithBiasAddAndRelu6<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kBiasAddWithLeakyRelu:
        conv2d(WithBiasAddAndLeakyRelu<T>(bias_add_args), context, input,
               filter, output);
        break;
      case FusedComputationType::kBiasAddWithElu:
        conv2d(WithBiasAddAndElu<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kFusedBatchNorm:
        conv2d(
            WithFusedBatchNorm<T>(fusion_args.epsilon, fused_batch_norm_args),
            context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithRelu:
        conv2d(WithFusedBatchNormAndRelu<T>(fusion_args.epsilon,
                                            fused_batch_norm_args),
               context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithRelu6:
        conv2d(WithFusedBatchNormAndRelu6<T>(fusion_args.epsilon,
                                             fused_batch_norm_args),
               context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithLeakyRelu:
        conv2d(WithFusedBatchNormAndLeakyRelu<T>(fusion_args.epsilon,
                                                 fused_batch_norm_args),
               context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithElu:
        conv2d(WithFusedBatchNormAndElu<T>(fusion_args.epsilon,
                                           fused_batch_norm_args),
               context, input, filter, output);
        break;
      default:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is unsupported"));
        break;
    }
  }
};

template <>
struct LaunchFusedConv2DOp<CPUDevice, int8>;

template <>
struct LaunchFusedConv2DOp<CPUDevice, qint8>;

#if GOOGLE_CUDA

inline int64_t ConvolveScratchSize() {
  static int64_t convolve_scratch_size = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );
  return convolve_scratch_size;
}

template <typename T>
struct LaunchFusedConv2DOp<GPUDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input_param,
                  const Tensor& filter, FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output) {
    OP_REQUIRES(
        context,
        params.data_format == FORMAT_NHWC || params.data_format == FORMAT_NCHW,
        errors::Unimplemented("Fused conv implementation only supports "
                              "NHWC and HCHW tensor formats for now."));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    OP_REQUIRES(
        context, use_cudnn,
        errors::Unimplemented("FusedConv2D for GPU is not currently supported "
                              "without cudnn"));

    bool is_supported_activation =
        fusion == FusedComputationType::kBiasAddWithRelu ||
        fusion == FusedComputationType::kBiasAddWithRelu6 ||
        fusion == FusedComputationType::kBiasAddWithElu ||
        fusion == FusedComputationType::kBiasAddWithLeakyRelu;
    OP_REQUIRES(
        context, is_supported_activation,
        errors::Unimplemented("FusedConv2D implementation only supports "
                              "fusing with `BiasAdd + Relu|Relu6|Elu|LeakyRlue`"
                              " for now."));

    Tensor input = input_param;

    const int64_t in_batch = GetTensorDim(input, params.data_format, 'N');
    int64_t in_rows = GetTensorDim(input, params.data_format, 'H');
    int64_t in_cols = GetTensorDim(input, params.data_format, 'W');
    const int64_t in_depths = GetTensorDim(input, params.data_format, 'C');

    const int64_t patch_rows = filter.dim_size(0);
    const int64_t patch_cols = filter.dim_size(1);
    const int64_t patch_depths = filter.dim_size(2);

    const int64_t out_batch = GetTensorDim(*output, params.data_format, 'N');
    const int64_t out_rows = GetTensorDim(*output, params.data_format, 'H');
    const int64_t out_cols = GetTensorDim(*output, params.data_format, 'W');
    const int64_t out_depths = GetTensorDim(*output, params.data_format, 'C');

    // Bias of the following dimensions: [ output_depth ]
    const Tensor& bias = context->input(2);
    OP_REQUIRES(context, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional",
                                        bias.shape().DebugString()));
    OP_REQUIRES(context, bias.dim_size(0) == out_depths,
                errors::InvalidArgument("bias depth must be equal to out depth",
                                        bias.shape().DebugString()));

    const int64_t common_padding_rows =
        std::min(dimensions.pad_rows_before, dimensions.pad_rows_after);
    const int64_t common_padding_cols =
        std::min(dimensions.pad_cols_before, dimensions.pad_cols_after);
    if (dimensions.pad_rows_before != dimensions.pad_rows_after ||
        dimensions.pad_cols_before != dimensions.pad_cols_after) {
      // cuDNN only supports padding the same amount on the left and right
      // sides, and on the top and bottom sides. So we manually create a new
      // padded input tensor such that we can pass it to cuDNN.

      // TODO(reedwm): In some cases, we can avoid an allocation even if the two
      // padding sides are different. For example, if the input is 2x2, the
      // filter is 1x1, the stride is 2, and the padding is (1, 0, 1, 0), the
      // result is equivalent to as if the padding is (1, 1, 1, 1). Changing the
      // padding in such a way would allow us to avoid the allocation.
      Tensor transformed_input;
      const int64_t padding_rows_diff =
          std::abs(dimensions.pad_rows_after - dimensions.pad_rows_before);
      const int64_t padding_cols_diff =
          std::abs(dimensions.pad_cols_after - dimensions.pad_cols_before);
      const int64_t new_in_rows = in_rows + padding_rows_diff;
      const int64_t new_in_cols = in_cols + padding_cols_diff;
      TensorShape transformed_input_shape;
      OP_REQUIRES_OK(context,
                     ShapeFromFormatWithStatus(
                         params.data_format, in_batch, new_in_rows, new_in_cols,
                         in_depths, &transformed_input_shape));
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     transformed_input_shape,
                                                     &transformed_input));
      const int64_t input_pad_top =
          dimensions.pad_rows_before - common_padding_rows;
      const int64_t input_pad_bottom =
          dimensions.pad_rows_after - common_padding_rows;
      const int64_t input_pad_left =
          dimensions.pad_cols_before - common_padding_cols;
      const int64_t input_pad_right =
          dimensions.pad_cols_after - common_padding_cols;
      bool in_bounds =
          FastBoundsCheck(input_pad_top, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_bottom, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_left, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_right, std::numeric_limits<int>::max());
      if (!in_bounds) {
        context->SetStatus(errors::InvalidArgument("Padding is too large."));
        return;
      }
      functor::PadInput<GPUDevice, T, int, 4>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(input_param.tensor<T, 4>()),
          {{static_cast<int>(input_pad_top), static_cast<int>(input_pad_left)}},
          {{static_cast<int>(input_pad_bottom),
            static_cast<int>(input_pad_right)}},
          To32Bit(transformed_input.tensor<T, 4>()), params.data_format, T{});
      input = transformed_input;
      in_rows = new_in_rows;
      in_cols = new_in_cols;
    }

    const bool compute_in_nhwc = DataTypeToEnum<T>::value == DT_HALF &&
                                 stream->GetCudaComputeCapability().IsAtLeast(
                                     se::CudaComputeCapability::VOLTA);
    if (!compute_in_nhwc && params.data_format == FORMAT_NHWC) {
      // Convert the input tensor from NHWC to NCHW.
      TensorShape nchw_shape;
      OP_REQUIRES_OK(
          context, ShapeFromFormatWithStatus(FORMAT_NCHW, in_batch, in_rows,
                                             in_cols, in_depths, &nchw_shape));
      if (in_depths > 1) {
        Tensor transformed_input;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());
        input = transformed_input;
      } else {
        // If depth <= 1, then just reshape.
        CHECK(input.CopyFrom(input, nchw_shape));  // Crash OK
      }
    }

    CHECK(common_padding_rows >= 0) << "Negative padding rows";  // Crash OK
    CHECK(common_padding_rows >= 0) << "Negative padding cols";  // Crash OK

    se::dnn::ActivationMode dnn_activation_mode;
    switch (fusion) {
      case FusedComputationType::kBiasAddWithRelu:
        dnn_activation_mode = se::dnn::ActivationMode::kRelu;
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        dnn_activation_mode = se::dnn::ActivationMode::kRelu6;
        break;
      case FusedComputationType::kBiasAddWithElu:
        dnn_activation_mode = se::dnn::ActivationMode::kElu;
        break;
      case FusedComputationType::kBiasAddWithLeakyRelu:
        dnn_activation_mode = se::dnn::ActivationMode::kLeakyRelu;
        break;
      default:
        LOG(FATAL) << "Unsupported fusion type";  // Crash OK
    }

    const TensorFormat compute_data_format =
        compute_in_nhwc ? FORMAT_NHWC : FORMAT_NCHW;
    constexpr auto kComputeInNHWC =
        std::make_tuple(se::dnn::DataLayout::kBatchYXDepth,
                        se::dnn::FilterLayout::kOutputYXInput);
    constexpr auto kComputeInNCHW =
        std::make_tuple(se::dnn::DataLayout::kBatchDepthYX,
                        se::dnn::FilterLayout::kOutputInputYX);
    se::dnn::DataLayout compute_data_layout;
    se::dnn::FilterLayout filter_layout;
    std::tie(compute_data_layout, filter_layout) =
        compute_in_nhwc ? kComputeInNHWC : kComputeInNCHW;

    se::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depths)
        .set_height(in_rows)
        .set_width(in_cols)
        .set_layout(compute_data_layout);
    se::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(patch_rows)
        .set_input_filter_width(patch_cols)
        .set_input_feature_map_count(patch_depths)
        .set_output_feature_map_count(filter.dim_size(3))
        .set_layout(filter_layout);
    se::dnn::BatchDescriptor bias_desc;
    bias_desc.set_count(1)
        .set_height(1)
        .set_width(1)
        .set_feature_map_count(out_depths)
        .set_layout(compute_data_layout);
    se::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_dilation_rate(dimensions.dilation_rows)
        .set_horizontal_dilation_rate(dimensions.dilation_cols)
        .set_vertical_filter_stride(dimensions.stride_rows)
        .set_horizontal_filter_stride(dimensions.stride_cols)
        .set_zero_padding_height(common_padding_rows)
        .set_zero_padding_width(common_padding_cols)
        .set_group_count(in_depths / patch_depths);
    se::dnn::BatchDescriptor output_desc;
    output_desc.set_count(out_batch)
        .set_height(out_rows)
        .set_width(out_cols)
        .set_feature_map_count(out_depths)
        .set_layout(compute_data_layout);

    Tensor transformed_filter;
    const auto transform_filter = [&](FilterTensorFormat dst_format) -> Status {
      VLOG(4) << "Transform filter tensor from " << ToString(FORMAT_HWIO)
              << " to " << ToString(dst_format);

      TensorShape dst_shape =
          dst_format == FORMAT_OIHW
              ? TensorShape({filter.dim_size(3), filter.dim_size(2),
                             filter.dim_size(0), filter.dim_size(1)})
              : TensorShape({filter.dim_size(3), filter.dim_size(0),
                             filter.dim_size(1), filter.dim_size(2)});

      TF_RETURN_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<T>::value, dst_shape, &transformed_filter));
      functor::TransformFilter<GPUDevice, T, int, 4>()(
          context->eigen_device<GPUDevice>(), dst_format,
          To32Bit(filter.tensor<T, 4>()),
          To32Bit(transformed_filter.tensor<T, 4>()));

      return OkStatus();
    };

    if (compute_in_nhwc) {
      OP_REQUIRES_OK(context, transform_filter(FORMAT_OHWI));
    } else {
      OP_REQUIRES_OK(context, transform_filter(FORMAT_OIHW));
    }

    Tensor transformed_output;
    if (!compute_in_nhwc && params.data_format == FORMAT_NHWC) {
      // Only allocate temporary memory when a layout transformation is needed.
      TensorShape transformed_output_shape;
      OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                  FORMAT_NCHW, out_batch, out_rows, out_cols,
                                  out_depths, &transformed_output_shape));
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     transformed_output_shape,
                                                     &transformed_output));
    } else {
      transformed_output = *output;
    }

    const auto tensor_on_device = [](const Tensor& t) -> se::DeviceMemory<T> {
      return AsDeviceMemory(t.template flat<T>().data(),
                            t.template flat<T>().size());
    };

    se::DeviceMemory<T> input_ptr = tensor_on_device(input);
    se::DeviceMemory<T> filter_ptr = tensor_on_device(transformed_filter);
    se::DeviceMemory<T> bias_ptr = tensor_on_device(bias);
    se::DeviceMemory<T> output_ptr = tensor_on_device(transformed_output);

    // We do not use side inputs, so we can safely pass nullptr.
    se::DeviceMemory<T> side_input_ptr =
        AsDeviceMemory(static_cast<T*>(nullptr), 0);

    constexpr double kConvScale = 1.0;
    constexpr double kSideInputScale = 0.0;
    double leakyrelu_alpha = fusion_args.leakyrelu_alpha;

    DataType dtype = input.dtype();
    ConvParameters conv_parameters = {
        stream->parent(),
        in_batch,                      // batch
        in_depths,                     // in_depths
        {{in_rows,                     // in_rows
          in_cols}},                   // in_cols
        compute_data_format,           // compute_data_format
        out_depths,                    // out_depths
        {{patch_rows,                  // filter_rows
          patch_cols,                  // filter_cols
          patch_depths}},              // filter_depths
        {{dimensions.dilation_rows,    // dilation_rows
          dimensions.dilation_cols}},  // dilation_cols
        {{dimensions.stride_rows,      // stride_rows
          dimensions.stride_cols}},    // stride_cols
        {{common_padding_rows,         // padding_rows
          common_padding_cols}},       // padding_cols
        dtype,                         // tensor datatype
        conv_desc.group_count(),
        ConvParameters::FusionInfo{kConvScale, kSideInputScale, leakyrelu_alpha,
                                   dnn_activation_mode,  // activation_mode
                                   /*is_contrib=*/false}};

    se::dnn::DataType element_type = se::dnn::ToDataType<T>::value;

    auto entry_or = AutotuneFusedConv<T>(
        cudnn_use_autotune, FusedConvAutotuneMap::GetInstance(),
        conv_parameters, context, input_desc, filter_desc, bias_desc,
        output_desc, conv_desc, dnn_activation_mode, kConvScale,
        kSideInputScale, leakyrelu_alpha, input_ptr, filter_ptr, output_ptr,
        bias_ptr, side_input_ptr, ConvolveScratchSize());
    OP_REQUIRES_OK(context, entry_or.status());
    auto autotune_entry = std::move(entry_or).value();

    DnnScratchAllocator scratch_allocator(ConvolveScratchSize(), context);
    Status cudnn_launch_status;
    if (!autotune_entry.is_algorithm_config()) {
      auto& runners = autotune_entry.GetOpRunners();
      se::dnn::FusedConvOp::Config config{se::dnn::ConvolutionKind::FORWARD,
                                          element_type,
                                          element_type,
                                          element_type,
                                          kConvScale,
                                          kSideInputScale,
                                          leakyrelu_alpha,
                                          input_desc,
                                          filter_desc,
                                          bias_desc,
                                          output_desc,
                                          conv_desc,
                                          dnn_activation_mode};
      auto primary_or = runners.primary->GetOrCreateRunner(config, stream);
      OP_REQUIRES_OK(context, primary_or.status());
      auto* primary = primary_or.value();

      const se::dnn::FusedConvRunner* no_scratch_fallback = nullptr;
      if (runners.no_scratch_fallback) {
        auto no_scratch_fallback_or =
            runners.no_scratch_fallback->GetOrCreateRunner(config, stream);
        OP_REQUIRES_OK(context, no_scratch_fallback_or.status());
        no_scratch_fallback = no_scratch_fallback_or.value();
      }

      auto runner_and_scratch_or =
          AllocateScratchOrFallback<se::dnn::FusedConvOp::Signature>(
              &scratch_allocator, primary, no_scratch_fallback);
      OP_REQUIRES_OK(context, runner_and_scratch_or.status());
      auto runner_and_scratch = std::move(runner_and_scratch_or).value();
      auto& runner =
          *std::get<const se::dnn::FusedConvRunner*>(runner_and_scratch);
      cudnn_launch_status = runner(
          stream, nullptr, std::get<se::DeviceMemoryBase>(runner_and_scratch),
          input_ptr, filter_ptr, side_input_ptr, bias_ptr, output_ptr);
    } else {
      auto dnn = stream->parent()->AsDnn();
      OP_REQUIRES(context, dnn != nullptr,
                  absl::InternalError("No DNN for stream."));
      cudnn_launch_status = dnn->FusedConvolveWithAlgorithm(
          stream, input_desc, input_ptr,    // input
          kConvScale,                       // input_scale
          filter_desc, filter_ptr,          // filter
          conv_desc,                        // conv
          side_input_ptr, kSideInputScale,  // side_input
          bias_desc, bias_ptr,              // bias
          dnn_activation_mode,              // activation
          output_desc, &output_ptr,         // output
          &scratch_allocator, autotune_entry.GetAlgorithmConfig(), nullptr);
    }

    OP_REQUIRES_OK(context, cudnn_launch_status);

    // Convert the output tensor back from NCHW to NHWC.
    if (!compute_in_nhwc && params.data_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
          output->tensor<T, 4>());
    }
  }
};

template <>
struct LaunchFusedConv2DOp<GPUDevice, int8>;

template <>
struct LaunchFusedConv2DOp<GPUDevice, qint8>;

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class FusedConv2DOp : public OpKernel {
 public:
  explicit FusedConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    cudnn_use_autotune_ = CudnnUseAutotune();

    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns;
    if (std::is_same<Device, CPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
          {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
          {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
          {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
          {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
          {FCT::kFusedBatchNormWithLeakyRelu, {"FusedBatchNorm", "LeakyRelu"}},
      };
    }

    // NOTE(ezhulenev): CuDNN `cudnnConvolutionBiasActivationForward` supports
    // identity activation function, it in theory should allow to fuse
    // convolution with BiasAdd, but in practice it doesn't work, cuDNN ignores
    // this parameter and always does Relu activation.
    if (std::is_same<Device, GPUDevice>::value) {
      if (std::is_same<T, int8>::value || std::is_same<T, qint8>::value) {
        patterns = {{FCT::kBiasAdd, {"BiasAdd"}},
                    {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}}};
      } else {
        patterns = {
            {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
            {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
            {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
            {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
        };
      }
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "Conv2D", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape;
    OP_REQUIRES_OK(
        context, ShapeFromFormatWithStatus(
                     params_.data_format, dimensions.batch, dimensions.out_rows,
                     dimensions.out_cols, dimensions.out_depth, &out_shape));

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "FusedConv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    LaunchFusedConv2DOp<Device, T>()(context, use_cudnn_, cudnn_use_autotune_,
                                     input, filter, fused_computation_,
                                     fused_computation_args_, params_,
                                     dimensions, output);
  }

 private:
  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  FusedConv2DOp(const FusedConv2DOp&) = delete;
  void operator=(const FusedConv2DOp&) = delete;
};

// Registration of the CPU implementations.
#define REGISTER_FUSED_CPU_CONV2D(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<CPUDevice, T>);

#if GOOGLE_CUDA

#define DECLARE_FUNCTOR_GPU_SPEC(T)                                     \
  template <>                                                           \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(               \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,         \
      typename TTypes<T, 4, int>::ConstTensor in,                       \
      typename TTypes<T, 4, int>::Tensor out);                          \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;         \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 4>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,   \
      const std::array<int, 2>& padding_left,                           \
      const std::array<int, 2>& padding_right,                          \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format, \
      const T& padding_value);                                          \
  extern template struct PadInput<GPUDevice, T, int, 4>

// Registration of the GPU implementations.
#define REGISTER_FUSED_GPU_CONV2D(T)                    \
  REGISTER_KERNEL_BUILDER(Name("_FusedConv2D")          \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("host_args"), \
                          FusedConv2DOp<GPUDevice, T>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_
