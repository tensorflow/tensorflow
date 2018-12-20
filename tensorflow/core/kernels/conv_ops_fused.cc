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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "cuda/include/cudnn.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

using CPUDevice = ::Eigen::ThreadPoolDevice;
using GPUDevice = ::Eigen::GpuDevice;

// Supported Conv2D fusions. Not all of them supported on all type of devices.
enum class FusedComputationType {
  // NOTE(ezhulenev): CuDNN `cudnnConvolutionBiasActivationForward` supports
  // identity activation function, it in theory should allow to fuse convolution
  // with BiasAdd, but in practice it doesn't work, cuDNN ignores this parameter
  // and always does Relu activation.
  kBiasAdd,                // CPU
  kBiasAddWithRelu,        // CPU and GPU
  kFusedBatchNorm,         // CPU only
  kFusedBatchNormWithRelu  // CPU only
};

// We have to pass around additional arguments for all possible fusion types.
struct FusedComputationArgs {
  float epsilon = 0.0;  // Used by `FusedBatchNorm` fusion only
};

template <typename Device, typename T>
struct LaunchFusedConv2DOp {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output);
};

// Type aliases for the unaligned tensors (tensor maps) used in output kernels.
template <typename T>
struct Unaligned {
  // There is no guarantee that the output block passed to the output kernel
  // will be aligned.

  using Tensor =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
                       Eigen::Unaligned>;

  using ConstTensor = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Unaligned>;
};

// Type alias for the tensor contraction output mapper.
template <typename Scalar, typename Index>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<Scalar, Index, Eigen::ColMajor>;

// Returns input expression without any transformations.
struct Identity {
  template <typename XprType>
  static auto apply(XprType expr) -> XprType {
    return expr;
  };
};

// Applies `Relu` to the passed input expression.
struct Relu {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0));
  };
};

// TensorContraction swaps lhs with rhs, and changes layout from RowMajor
// (default in Tensorflow) to ColMajor (preferred in Eigen), and computes matmul
// using these tensors.
//
// TensorContraction output matrix (before reshape) has a ColMajor layout, and
// has dimensions:
//  - rows: output_channels
//  - cols: all other dimensions
//
// First element in every column is:
//   [batch ??, height ??, width ??, out_channel = i]
//
// We do not know what are the values of the 'batch', 'height', and 'width' here
// (if we know original dimensions, they can be computed from 'j').
//
// Each column of an output block is a continuous slice along the output channel
// dimension, so we can use it to efficiently compute any transformation that
// depends only on a channel value (e.g. add channel bias).

// Output kernel that fuses BiasAdd operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct BiasAddOutputKernel {
  explicit BiasAddOutputKernel(const T* bias_data) : bias_data(bias_data) {}

  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, Index>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename Unaligned<T>::ConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename Unaligned<T>::Tensor output(output_base, num_rows);
      const auto expr = output + bias;
      output = Activation::template apply<decltype(expr)>(expr);
    }
  }

 private:
  const T* bias_data;
};

// Output kernel that fuses FusedBatchNorm operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct FusedBatchNormOutputKernel {
  FusedBatchNormOutputKernel(T epsilon, const T* scaling_factor_data,
                             const T* offset_data, const T* estimated_mean_data)
      : epsilon(epsilon),
        scaling_factor_data(scaling_factor_data),
        offset_data(offset_data),
        estimated_mean_data(estimated_mean_data) {}

  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, Index>& output_mapper,
      const Eigen::TensorContractionParams& params, Index i, Index j,
      Index num_rows, Index num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename Unaligned<T>::ConstTensor scaling_factor(scaling_factor_base,
                                                      num_rows);
    typename Unaligned<T>::ConstTensor offset(offset_base, num_rows);
    typename Unaligned<T>::ConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename Unaligned<T>::Tensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = Activation::template apply<decltype(shifted)>(shifted);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
};

// Type aliases for the output kernels, purely for the sake of better launch
// dispatching code readability.
template <typename T>
using WithBiasAdd = BiasAddOutputKernel<T>;
template <typename T>
using WithBiasAddAndRelu = BiasAddOutputKernel<T, Relu>;
template <typename T>
using WithFusedBatchNorm = FusedBatchNormOutputKernel<T>;
template <typename T>
using WithFusedBatchNormAndRelu = FusedBatchNormOutputKernel<T, Relu>;

// This is CPU-only implementation that uses Eigen contraction output kernels.
//
// Dispatch 2D convolution to the appropriate primitive operation:
//   (1) MatMul for the case of 1x1 convolution.
//   (2) MatMul for the case when filter size equals to the input size.
//   (3) General spatial 2D convolution for all other cases.
template <typename T>
class LaunchFusedConv2DWithOutputKernel {
 public:
  LaunchFusedConv2DWithOutputKernel(int row_stride, int col_stride,      //
                                    int row_dilation, int col_dilation,  //
                                    Padding padding)
      : row_stride_(row_stride),
        col_stride_(col_stride),
        row_dilation_(row_dilation),
        col_dilation_(col_dilation),
        padding_(padding) {}

  template <typename OutputKernel>
  void operator()(const OutputKernel& output_kernel, OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter, Tensor* output) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        row_stride_ == 1 && col_stride_ == 1) {
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair, output_kernel);

    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation_ == 1 &&
               col_dilation_ == 1 && padding_ == VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair,
          output_kernel);

    } else {
      functor::SpatialConvolution<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
          input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_, col_stride_,
          row_dilation_, col_dilation_, BrainPadding2EigenPadding(padding_),
          output_kernel);
    }
  }

 private:
  int row_stride_;
  int col_stride_;
  int row_dilation_;
  int col_dilation_;
  const Padding padding_;
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

    BiasAddArgs bias_add;
    FusedBatchNormArgs fused_batch_norm;

    LaunchFusedConv2DWithOutputKernel<T> conv2d(
        dimensions.stride_rows, dimensions.stride_cols,
        dimensions.dilation_rows, dimensions.dilation_cols, params.padding);

    switch (fusion) {
      case FusedComputationType::kBiasAdd:
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add));
        conv2d(WithBiasAdd<T>(bias_add.bias_add_data), context, input, filter,
               output);
        break;

      case FusedComputationType::kBiasAddWithRelu:
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add));
        conv2d(WithBiasAddAndRelu<T>(bias_add.bias_add_data), context, input,
               filter, output);
        break;

      case FusedComputationType::kFusedBatchNorm:
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                              &fused_batch_norm));
        conv2d(WithFusedBatchNorm<T>(fusion_args.epsilon,
                                     fused_batch_norm.scaling_factor.data(),
                                     fused_batch_norm.offset_data,
                                     fused_batch_norm.estimated_mean_data),
               context, input, filter, output);
        break;

      case FusedComputationType::kFusedBatchNormWithRelu:
        OP_REQUIRES_OK(context,
                       InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                              &fused_batch_norm));
        conv2d(WithFusedBatchNormAndRelu<T>(
                   fusion_args.epsilon, fused_batch_norm.scaling_factor.data(),
                   fused_batch_norm.offset_data,
                   fused_batch_norm.estimated_mean_data),
               context, input, filter, output);
        break;
    }
  }

 private:
  struct BiasAddArgs {
    const T* bias_add_data = nullptr;
  };

  struct FusedBatchNormArgs {
    const T* scale_data = nullptr;
    const T* offset_data = nullptr;
    const T* estimated_mean_data = nullptr;
    const T* estimated_variance_data = nullptr;

    // Precomputed expression:
    //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
    Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor;
  };

#define TF_REQUIRES(EXP, STATUS) \
  if (!TF_PREDICT_TRUE(EXP)) return (STATUS)

  void InitDataPtr(const Tensor& tensor, const T** ptr) const {
    *ptr = reinterpret_cast<const T*>(tensor.tensor_data().data());
  }

  Status InitBiasAddArgs(OpKernelContext* context, BiasAddArgs* args) const {
    // Bias of the following dimensions: [ output_depth ]
    const Tensor& bias = context->input(2);

    TF_REQUIRES(bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional",
                                        bias.shape().DebugString()));

    InitDataPtr(bias, &args->bias_add_data);

    return Status::OK();
  }

  Status InitFusedBatchNormArgs(OpKernelContext* context, float epsilon,
                                FusedBatchNormArgs* args) const {
    const Tensor& scale = context->input(2);
    const Tensor& offset = context->input(3);
    const Tensor& estimated_mean = context->input(4);
    const Tensor& estimated_variance = context->input(5);

    TF_REQUIRES(scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    TF_REQUIRES(offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    TF_REQUIRES(estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    TF_REQUIRES(
        estimated_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_variance.shape().DebugString()));

    InitDataPtr(scale, &args->scale_data);
    InitDataPtr(offset, &args->offset_data);
    InitDataPtr(estimated_mean, &args->estimated_mean_data);
    InitDataPtr(estimated_variance, &args->estimated_variance_data);

    // Precompute scaling factor once for all output blocks (kernels).
    args->scaling_factor =
        (estimated_variance.flat<T>() + static_cast<T>(epsilon)).rsqrt() *
        scale.flat<T>();

    return Status::OK();
  }

#undef TF_REQUIRES
};

#if GOOGLE_CUDA

// Encapsulate the default shape information that is used by the convolution
// operation, and add an activation mode for the fusion.
class FusedConvParameters : public ConvParameters {
 public:
  FusedConvParameters(const ConvParameters& base,
                      const se::dnn::ActivationMode activation_mode)
      : ConvParameters(base), activation_mode_(activation_mode) {}

  string ToString() const {
    return absl::StrCat(ConvParameters::ToString(), ", ", activation_mode_);
  }

 private:
  friend bool operator==(const FusedConvParameters& lhs,
                         const FusedConvParameters& rhs);

  using ParameterDataType =
      std::tuple<ConvParameters::ParameterDataType, se::dnn::ActivationMode>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(ConvParameters::get_data_as_tuple(),
                           activation_mode_);
  }

  se::dnn::ActivationMode activation_mode_;
};

bool operator==(const FusedConvParameters& lhs,
                const FusedConvParameters& rhs) {
  return lhs.get_data_as_tuple() == rhs.get_data_as_tuple();
}

bool operator!=(const FusedConvParameters& lhs,
                const FusedConvParameters& rhs) {
  return !(lhs == rhs);
}

// A dummy type to group forward convolution autotune results together.
struct FusedConvAutoTuneGroup {
  static string name() { return "FusedConv"; }
};

using AutoTuneFusedConv =
    AutoTuneSingleton<FusedConvAutoTuneGroup, FusedConvParameters,
                      se::dnn::AlgorithmConfig>;

int64 ConvolveScratchSize() {
  static int64 convolve_scratch_size = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );
  return convolve_scratch_size;
}

// Finds the best convolutiun algorithm for the given ConvLaunch (cuda
// convolution on the stream) and parameters, by running all possible
// algorithms and measuring execution time.
// TODO(ezhulenev): Move it to conv_ops_gpu.h and share with conv_ops.cc.
template <typename T, typename ConvLaunch>
Status FindBestConvolveAlgorithm(const FusedConvParameters& params,
                                 const ConvLaunch launch,
                                 OpKernelContext* context, se::Stream* stream,
                                 se::dnn::AlgorithmConfig* algorithm_config) {
  // Check if we already have an algorithm selected for the given parameters.
  if (AutoTuneFusedConv::GetInstance()->Find(params, algorithm_config)) {
    return Status::OK();
  }

  // Find all candidate algorithms.
  std::vector<se::dnn::AlgorithmDesc> algorithms;
  if (!stream->parent()->GetConvolveAlgorithms(
          params.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()),
          &algorithms)) {
    return errors::Unknown(
        "Failed to get convolution algorithm. This is probably "
        "because cuDNN failed to initialize, so try looking to "
        "see if a warning log message was printed above.");
  }

  se::dnn::ProfileResult best_result;
  se::dnn::ProfileResult best_result_no_scratch;

  for (auto profile_algorithm : algorithms) {
    DnnScratchAllocator scratch_allocator(ConvolveScratchSize(), context);
    se::dnn::ProfileResult profile_result;

    bool cudnn_launch_status =
        launch(se::dnn::AlgorithmConfig(profile_algorithm), &scratch_allocator,
               &profile_result);

    if (cudnn_launch_status && profile_result.is_valid()) {
      if (profile_result.elapsed_time_in_ms() <
          best_result.elapsed_time_in_ms()) {
        best_result = profile_result;
      }
      if (scratch_allocator.TotalByteSize() == 0 &&
          profile_result.elapsed_time_in_ms() <
              best_result_no_scratch.elapsed_time_in_ms()) {
        best_result_no_scratch = profile_result;
      }
    }
  }

  if (!best_result.is_valid() && !best_result_no_scratch.is_valid()) {
    return errors::NotFound("No algorithm worked!");
  }
  if (best_result.is_valid()) {
    algorithm_config->set_algorithm(best_result.algorithm());
  }
  if (best_result_no_scratch.is_valid()) {
    algorithm_config->set_algorithm_no_scratch(
        best_result_no_scratch.algorithm());
  }

  AutoTuneFusedConv::GetInstance()->Insert(params, *algorithm_config);
  return Status::OK();
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

    OP_REQUIRES(
        context, fusion == FusedComputationType::kBiasAddWithRelu,
        errors::Unimplemented("FusedConv2D implementation only supports "
                              "fusing with `BiasAdd + Relu` for now."));

    Tensor input = input_param;

    const int64 in_batch = GetTensorDim(input, params.data_format, 'N');
    int64 in_rows = GetTensorDim(input, params.data_format, 'H');
    int64 in_cols = GetTensorDim(input, params.data_format, 'W');
    const int64 in_depths = GetTensorDim(input, params.data_format, 'C');

    const int64 patch_rows = filter.dim_size(0);
    const int64 patch_cols = filter.dim_size(1);
    const int64 patch_depths = filter.dim_size(2);

    int64 padding_rows = 0;
    int64 padding_cols = 0;
    const int64 out_batch = GetTensorDim(*output, params.data_format, 'N');
    const int64 out_rows = GetTensorDim(*output, params.data_format, 'H');
    const int64 out_cols = GetTensorDim(*output, params.data_format, 'W');
    const int64 out_depths = GetTensorDim(*output, params.data_format, 'C');

    // Bias of the following dimensions: [ output_depth ]
    const Tensor& bias = context->input(2);
    OP_REQUIRES(context, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional",
                                        bias.shape().DebugString()));
    OP_REQUIRES(context, bias.dim_size(0) == out_depths,
                errors::InvalidArgument("bias depth must be equal to out depth",
                                        bias.shape().DebugString()));

    if (params.padding == SAME) {
      // Total padding on rows and cols is
      // Pr = (R' - 1) * S + (Kr - 1) * Dr + 1 - R
      // Pc = (C' - 1) * S + (Kc - 1) * Dc + 1 - C
      // where (R', C') are output dimensions, (R, C) are input dimensions, S
      // is stride, (Dr, Dc) are dilations, (Kr, Kc) are filter dimensions.
      // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
      // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
      // we pad more on the right and bottom than on the top and left.
      padding_rows = std::max<int>(
          0, (out_rows - 1) * dimensions.stride_rows +
                 (patch_rows - 1) * dimensions.dilation_rows + 1 - in_rows);
      padding_cols = std::max<int>(
          0, (out_cols - 1) * dimensions.stride_cols +
                 (patch_cols - 1) * dimensions.dilation_cols + 1 - in_cols);
      const bool rows_odd = (padding_rows % 2 != 0);
      const bool cols_odd = (padding_cols % 2 != 0);
      if (rows_odd || cols_odd) {
        Tensor transformed_input;
        int64 new_in_rows = in_rows + rows_odd;
        int64 new_in_cols = in_cols + cols_odd;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(
                           DataTypeToEnum<T>::value,
                           ShapeFromFormat(params.data_format, in_batch,
                                           new_in_rows, new_in_cols, in_depths),
                           &transformed_input));

        functor::PadInput<GPUDevice, T, int, 4>()(
            context->eigen_device<GPUDevice>(),
            To32Bit(input_param.tensor<T, 4>()), {{0, 0}},
            {{rows_odd, cols_odd}}, To32Bit(transformed_input.tensor<T, 4>()),
            params.data_format);

        input = transformed_input;
        in_rows = new_in_rows;
        in_cols = new_in_cols;
      }
    }

    if (params.data_format == FORMAT_NHWC) {
      // Convert the input tensor from NHWC to NCHW.
      TensorShape nchw_shape =
          ShapeFromFormat(FORMAT_NCHW, in_batch, in_rows, in_cols, in_depths);
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

    CHECK(padding_rows >= 0) << "Negative padding rows";  // Crash OK
    CHECK(padding_cols >= 0) << "Negative padding cols";  // Crash OK

    se::dnn::ActivationMode dnn_activation_mode;
    switch (fusion) {
      case FusedComputationType::kBiasAddWithRelu:
        dnn_activation_mode = se::dnn::ActivationMode::kRelu;
        break;
      default:
        LOG(FATAL) << "Unsupported fusion type";  // Crash OK
    }

    se::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depths)
        .set_height(in_rows)
        .set_width(in_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(patch_rows)
        .set_input_filter_width(patch_cols)
        .set_input_feature_map_count(patch_depths)
        .set_output_feature_map_count(filter.dim_size(3));
    se::dnn::BatchDescriptor bias_desc;
    bias_desc.set_count(1)
        .set_height(1)
        .set_width(1)
        .set_feature_map_count(out_depths)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_dilation_rate(dimensions.dilation_rows)
        .set_horizontal_dilation_rate(dimensions.dilation_cols)
        .set_vertical_filter_stride(dimensions.stride_rows)
        .set_horizontal_filter_stride(dimensions.stride_cols)
        .set_zero_padding_height(padding_rows / 2)
        .set_zero_padding_width(padding_cols / 2)
        .set_group_count(in_depths / patch_depths);
    se::dnn::BatchDescriptor output_desc;
    output_desc.set_count(out_batch)
        .set_height(out_rows)
        .set_width(out_cols)
        .set_feature_map_count(out_depths)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    Tensor transformed_filter;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({filter.dim_size(3), filter.dim_size(2),
                                    filter.dim_size(0), filter.dim_size(1)}),
                       &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        context->eigen_device<GPUDevice>(), FORMAT_OIHW,
        To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    Tensor transformed_output;
    if (params.data_format == FORMAT_NHWC) {
      // Only allocate temporary memory when a layout transformation is needed.
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         ShapeFromFormat(FORMAT_NCHW, out_batch, out_rows,
                                         out_cols, out_depths),
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

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = input.dtype();
    FusedConvParameters conv_parameters = {
        {
            in_batch,                      // batch
            in_depths,                     // in_depths
            {{in_rows,                     // in_rows
              in_cols}},                   // in_cols
            FORMAT_NCHW,                   // compute_data_format
            out_depths,                    // out_depths
            {{patch_rows,                  // filter_rows
              patch_cols,                  // filter_cols
              patch_depths}},              // filter_depths
            {{dimensions.dilation_rows,    // dilation_rows
              dimensions.dilation_cols}},  // dilation_cols
            {{dimensions.stride_rows,      // stride_rows
              dimensions.stride_cols}},    // stride_cols
            {{padding_rows,                // padding_rows
              padding_cols}},              // padding_cols
            dtype,                         // tensor datatype
            device_id,                     // device_id
        },
        dnn_activation_mode  // activation_mode
    };

    // Launch fused convolution with given parameters and scratch allocator.
    // Record profile result into `profile_result` if it's not nullptr.
    const auto launch = [&](se::dnn::AlgorithmConfig algorithm_config,
                            DnnScratchAllocator* scratch_allocator,
                            se::dnn::ProfileResult* profile_result) -> bool {
      return stream
          ->ThenFusedConvolveWithAlgorithm(
              input_desc, input_ptr,                     // input
              /*conv_input_scale=*/1.0,                  // input_scale
              filter_desc, filter_ptr,                   // filter
              conv_desc,                                 // conv
              side_input_ptr, /*side_input_scale=*/0.0,  // side_input
              bias_desc, bias_ptr,                       // bias
              dnn_activation_mode,                       // activation
              output_desc, &output_ptr,                  // output
              scratch_allocator, algorithm_config, profile_result)
          .ok();
    };

    se::dnn::AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune) {
      OP_REQUIRES_OK(context, FindBestConvolveAlgorithm<T>(
                                  conv_parameters, launch, context, stream,
                                  &algorithm_config));
    }

    DnnScratchAllocator scratch_allocator(ConvolveScratchSize(), context);
    bool cudnn_launch_status = launch(algorithm_config, &scratch_allocator,
                                      /*profile_result=*/nullptr);
    OP_REQUIRES(
        context, cudnn_launch_status,
        errors::Internal(absl::Substitute(
            "cuDNN launch failure: input shape($0) filter shape($1)",
            input.shape().DebugString(), filter.shape().DebugString())));

    // Convert the output tensor back from NCHW to NHWC.
    if (params.data_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
          output->tensor<T, 4>());
    }
  }
};

#endif  // GOOGLE_CUDA

}  // namespace

template <typename Device, typename T>
class FusedConv2DOp : public OpKernel {
 public:
  explicit FusedConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();

    // 'fused_ops' and 'num_args' attributes are specified by the Grappler
    // Remapper optimizer (see grappler/optimizers/remapper.cc).

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused Conv2D must have at least one fused op."));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));

    // TODO(ezhulenev): Add support for fusion element-wise op chains defined
    // at runtime, e.g. Relu+Sqrt+Tanh+etc.

    // Match combination of fused ops to one of the supported fusions.
    if (FusedOpsMatchAndSupportedOnDevice(fused_ops, {"BiasAdd"},
                                          /*cpu_only=*/true)) {
      fused_computation_ = FusedComputationType::kBiasAdd;
    } else if (FusedOpsMatchAndSupportedOnDevice(fused_ops, {"BiasAdd", "Relu"},
                                                 /*cpu_only=*/false)) {
      fused_computation_ = FusedComputationType::kBiasAddWithRelu;
    } else if (FusedOpsMatchAndSupportedOnDevice(fused_ops, {"FusedBatchNorm"},
                                                 /*cpu_only=*/true)) {
      fused_computation_ = FusedComputationType::kFusedBatchNorm;
    } else if (FusedOpsMatchAndSupportedOnDevice(fused_ops,
                                                 {"FusedBatchNorm", "Relu"},
                                                 /*cpu_only=*/true)) {
      fused_computation_ = FusedComputationType::kFusedBatchNormWithRelu;
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        absl::StrJoin(fused_ops, ","), "]"));
    }

    // Depending on a picked fusion type validate fusion-specific arguments.

    if (fused_computation_ == FusedComputationType::kBiasAdd ||
        fused_computation_ == FusedComputationType::kBiasAddWithRelu) {
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    }

    if (fused_computation_ == FusedComputationType::kFusedBatchNorm ||
        fused_computation_ == FusedComputationType::kFusedBatchNormWithRelu) {
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument("Fused FusedBatchNorm must have four extra "
                                  "arguments: scale, offset, mean, variance."));
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    }
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

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

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

    FusedComputationArgs args;
    args.epsilon = epsilon_;

    LaunchFusedConv2DOp<Device, T>()(context, use_cudnn_, cudnn_use_autotune_,
                                     input, filter, fused_computation_, args,
                                     params_, dimensions, output);
  }

 private:
  bool FusedOpsMatchAndSupportedOnDevice(const std::vector<string>& fused_ops,
                                         const std::vector<string>& expected,
                                         bool cpu_only) const {
    if (std::is_same<Device, GPUDevice>::value && cpu_only) {
      return false;
    }
    return fused_ops == expected;
  }

  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  FusedComputationType fused_computation_;

  float epsilon_;  // Used only in FusedBatchNorm fusion

  TF_DISALLOW_COPY_AND_ASSIGN(FusedConv2DOp);
};

// Registration of the CPU implementations.
#define REGISTER_FUSED_CPU_CONV2D(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<CPUDevice, T>);

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
// TODO(b/119765980): Upgrade upstream Eigen to set `m_can_use_xsmm=false` for
// contractions with non-default contraction output kernels.
#if !defined(USE_GEMM_FOR_CONV) && !defined(EIGEN_USE_LIBXSMM)
TF_CALL_float(REGISTER_FUSED_CPU_CONV2D);
TF_CALL_double(REGISTER_FUSED_CPU_CONV2D);
#endif  // !USE_GEMM_FOR_CONV

#undef REGISTER_FUSED_CPU_CONV2D

#if GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,          \
      typename TTypes<T, 4, int>::ConstTensor in,                        \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;          \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_FUSED_GPU_CONV2D(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<GPUDevice, T>);

TF_CALL_float(REGISTER_FUSED_GPU_CONV2D);
TF_CALL_double(REGISTER_FUSED_GPU_CONV2D);

#undef REGISTER_FUSED_GPU_CONV2D

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
