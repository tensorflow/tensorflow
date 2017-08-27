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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/fused_conv/kernels/fused_conv2d_bias_activation_op.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/activation_mode.h"
#endif  // GOOGLE_CUDA
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchConvOp;

template <typename Device, typename T>
class FusedConv2DBiasActivationOp : public OpKernel {
 public:
  explicit FusedConv2DBiasActivationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context,
                (data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW),
                errors::InvalidArgument("Current implementation only supports "
                                        "NHWC and NCHW data formats."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(strides_, data_format_, 'N') == 1 &&
         GetTensorDim(strides_, data_format_, 'C') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    string activation_mode_str;
    OP_REQUIRES_OK(context,
                   context->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(context, GetActivationModeFromString(activation_mode_str,
                                                        &activation_mode_));
    OP_REQUIRES(context, activation_mode_ == ActivationMode::RELU,
                errors::InvalidArgument("Current implementation only supports "
                                        "relu as the activation mode."));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is one of the following shapes:
    // [ batch, in_rows, in_cols, in_depth ] (for NHWC data format)
    // [ batch, in_depth, in_rows, in_cols ] (for NCHW data format)
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth ]
    const Tensor& filter = context->input(1);

    // Input bias is a 1-D tensor the size of the last
    // dimension of Output tensor
    const Tensor& bias = context->input(2);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // Bias should be a 1-D tensor.
    OP_REQUIRES(context, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional: ",
                                        bias.shape().DebugString()));

    for (int i = 0; i < 4; i++) {
      OP_REQUIRES(context,
                  FastBoundsCheck(filter.dim_size(i),
                                  std::numeric_limits<int32>::max()),
                  errors::InvalidArgument("filter dimension too large"));
      OP_REQUIRES(
          context,
          FastBoundsCheck(input.dim_size(i), std::numeric_limits<int32>::max()),
          errors::InvalidArgument("input dimension too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int32 out_depth = static_cast<int32>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    const int32 input_rows = static_cast<int32>(input_rows_raw);
    const int32 filter_rows = static_cast<int32>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    const int32 input_cols = static_cast<int32>(input_cols_raw);
    const int32 filter_cols = static_cast<int32>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    const int32 batch = static_cast<int32>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int32 stride_rows =
        static_cast<int32>(GetTensorDim(strides_, data_format_, 'H'));
    const int32 stride_cols =
        static_cast<int32>(GetTensorDim(strides_, data_format_, 'W'));
    const int32 bias_size = static_cast<int32>(bias.dim_size(0));

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));
    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Bias size should be the same as the size of the channel dimension of
    // output.
    OP_REQUIRES(context, bias_size == out_depth,
                errors::InvalidArgument(
                    "bias size should equal the channel "
                    "dimension size of output. bias shape: ",
                    bias.shape().DebugString() +
                        ", output shape: " + output->shape().DebugString()));

    VLOG(2) << "FusedConv2DBiasActivation: in_depth = " << in_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", bias_size = " << bias_size << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    launcher_.launch(context, cudnn_use_autotune_, input, filter, stride_rows,
                     stride_cols, bias, activation_mode_,
                     BrainPadding2EigenPadding(padding_), data_format_, output);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  ActivationMode activation_mode_;
  TensorFormat data_format_;
  LaunchFusedConv2DBiasActivationOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedConv2DBiasActivationOp);
};

#if GOOGLE_CUDA
namespace dnn = ::perftools::gputools::dnn;

dnn::ActivationMode BrainActivationMode2CudnnActivationMode(
    ActivationMode activation_mode) {
  switch (activation_mode) {
    case ActivationMode::SIGMOID:
      return dnn::ActivationMode::kSigmoid;
    case ActivationMode::RELU:
      return dnn::ActivationMode::kRelu;
    case ActivationMode::RELUX:
      return dnn::ActivationMode::kReluX;
    case ActivationMode::RELU6:
      return dnn::ActivationMode::kRelu6;
    case ActivationMode::TANH:
      return dnn::ActivationMode::kTanh;
    case ActivationMode::BANDPASS:
      return dnn::ActivationMode::kBandPass;
  }
  // Prevent compiler warning about missing return
  return dnn::ActivationMode::kRelu;
}

// A dummy type to group forward convolution autotune results together.
struct ConvBiasActivationAutoTuneGroup {
  static string name() { return "ConvBiasActivation"; }
};
typedef AutoTuneSingleton<ConvBiasActivationAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>
    AutoTuneConvBiasActivation;

template <typename T>
void LaunchFusedConv2DBiasActivationOp<GPUDevice, T>::launch(
    OpKernelContext* ctx, bool cudnn_use_autotune, const Tensor& input_param,
    const Tensor& filter, int32 row_stride, int32 col_stride,
    const Tensor& bias, const ActivationMode& activation_mode,
    const Eigen::PaddingType& padding, TensorFormat data_format,
    Tensor* output) {
  using perftools::gputools::dnn::AlgorithmConfig;
  using perftools::gputools::dnn::AlgorithmType;
  using perftools::gputools::dnn::ProfileResult;
  using perftools::gputools::dnn::kDefaultAlgorithm;
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  Tensor input = input_param;

  perftools::gputools::dnn::ActivationMode cudnn_activation_mode =
      BrainActivationMode2CudnnActivationMode(activation_mode);

  // TODO(yangzihao): refactor all the complicated/duplicated code in regular
  // conv ops to a shared conv utility.
  int32 padding_rows = 0;
  int32 padding_cols = 0;
  const int64 in_batch = GetTensorDim(input, data_format, 'N');
  int64 in_rows = GetTensorDim(input, data_format, 'H');
  int64 in_cols = GetTensorDim(input, data_format, 'W');
  const int64 in_depths = GetTensorDim(input, data_format, 'C');
  const int64 out_batch = GetTensorDim(*output, data_format, 'N');
  const int64 out_rows = GetTensorDim(*output, data_format, 'H');
  const int64 out_cols = GetTensorDim(*output, data_format, 'W');
  const int64 out_depths = GetTensorDim(*output, data_format, 'C');
  const int64 patch_rows = filter.dim_size(0);
  const int64 patch_cols = filter.dim_size(1);
  if (padding == Eigen::PADDING_SAME) {
    // Total padding on rows and cols is
    // Pr = (R' - 1) * S + Kr - R
    // Pc = (C' - 1) * S + Kc - C
    // where (R', C') are output dimensions, (R, C) are input dimensions, S
    // is stride, (Kr, Kc) are filter dimensions.
    // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
    // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
    // we pad more on the right and bottom than on the top and left.
    padding_rows =
        std::max<int32>(0, (out_rows - 1) * row_stride + patch_rows - in_rows);
    padding_cols =
        std::max<int32>(0, (out_cols - 1) * col_stride + patch_cols - in_cols);
    const int rows_parity = padding_rows & 1;
    const int cols_parity = padding_cols & 1;
    if ((rows_parity | cols_parity) != 0) {
      Tensor transformed_input;
      int64 new_in_rows = in_rows + rows_parity;
      int64 new_in_cols = in_cols + cols_parity;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(DataTypeToEnum<T>::value,
                             ShapeFromFormat(data_format, in_batch, new_in_rows,
                                             new_in_cols, in_depths),
                             &transformed_input));

      functor::PadInput<GPUDevice, T, int, 4>()(
          ctx->eigen_device<GPUDevice>(), To32Bit(input_param.tensor<T, 4>()),
          {{0, 0}}, {{rows_parity, cols_parity}},
          To32Bit(transformed_input.tensor<T, 4>()), data_format);

      input = transformed_input;
      in_rows = new_in_rows;
      in_cols = new_in_cols;
    }
  }

  if (data_format == FORMAT_NHWC) {
    // Convert the input tensor from NHWC to NCHW.
    TensorShape nchw_shape =
        ShapeFromFormat(FORMAT_NCHW, in_batch, in_rows, in_cols, in_depths);
    if (in_depths > 1) {
      Tensor transformed_input;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             nchw_shape, &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
      input = transformed_input;
    } else {
      // If depth <= 1, then just reshape.
      CHECK(input.CopyFrom(input, nchw_shape));
    }
  }

  CHECK(padding_rows >= 0 && padding_cols >= 0)
      << "Negative row or col paddings: (" << padding_rows << ", "
      << padding_cols << ")";
  perftools::gputools::dnn::BatchDescriptor input_desc;
  input_desc.set_count(in_batch)
      .set_feature_map_count(in_depths)
      .set_height(in_rows)
      .set_width(in_cols)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
  perftools::gputools::dnn::BatchDescriptor output_desc;
  output_desc.set_count(out_batch)
      .set_height(out_rows)
      .set_width(out_cols)
      .set_feature_map_count(out_depths)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
  perftools::gputools::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(filter.dim_size(0))
      .set_input_filter_width(filter.dim_size(1))
      .set_input_feature_map_count(filter.dim_size(2))
      .set_output_feature_map_count(filter.dim_size(3));
  perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_filter_stride(row_stride)
      .set_horizontal_filter_stride(col_stride)
      .set_zero_padding_height(padding_rows / 2)
      .set_zero_padding_width(padding_cols / 2);

  // Shuffles a filter tensor from:
  //   [<spatial_dims>, in, out]
  // to:
  //   [out, in, <spatial_dims>]
  // TODO(yangzihao): Support a data layout tag for the filter weights, and only
  // do the transform if the weights are not already in the correct layout.
  Tensor transformed_filter;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DataTypeToEnum<T>::value,
                          TensorShape({filter.dim_size(3), filter.dim_size(2),
                                       filter.dim_size(0), filter.dim_size(1)}),
                          &transformed_filter));

  functor::TransformFilter<GPUDevice, T, int, 4>()(
      ctx->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 4>()),
      To32Bit(transformed_filter.tensor<T, 4>()));

  Tensor transformed_output;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                              ShapeFromFormat(FORMAT_NCHW, out_batch, out_rows,
                                              out_cols, out_depths),
                              &transformed_output));

  auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
  auto filter_ptr =
      AsDeviceMemory(transformed_filter.template flat<T>().data(),
                     transformed_filter.template flat<T>().size());
  auto output_ptr =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());

  auto bias_ptr = AsDeviceMemory(bias.template flat<T>().data(),
                                 bias.template flat<T>().size());

  static int64 ConvolveScratchSize = GetCudnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  int device_id = stream->parent()->device_ordinal();
  DataType dtype = input.dtype();
  ConvParameters conv_parameters = {
      in_batch,
      in_depths,
      {{in_rows, in_cols}},
      out_depths,
      {{patch_rows, patch_cols}},
      {{row_stride, col_stride}},
      {{padding_rows, padding_cols}},
      dtype,
      device_id,
  };

  AlgorithmConfig algorithm_config;
  if (cudnn_use_autotune && !AutoTuneConvBiasActivation::GetInstance()->Find(
                                conv_parameters, &algorithm_config)) {
    std::vector<AlgorithmType> algorithms;
    CHECK(stream->parent()->GetConvolveAlgorithms(
        conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms));
    ProfileResult best_result;
    ProfileResult best_result_no_scratch;
    for (auto profile_algorithm : algorithms) {
      // TODO(zhengxq): profile each algorithm multiple times to better
      // accuracy.
      CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
      ProfileResult profile_result;
      bool cudnn_launch_status =
          stream
              ->ThenConvolveWithAlgorithm(
                  input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
                  bias_ptr, cudnn_activation_mode, output_desc, &output_ptr,
                  &scratch_allocator, AlgorithmConfig(profile_algorithm),
                  &profile_result)
              .ok();
      if (cudnn_launch_status) {
        if (profile_result.is_valid()) {
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
    }
    OP_REQUIRES(ctx,
                best_result.is_valid() || best_result_no_scratch.is_valid(),
                errors::NotFound("No algorithm worked!"));
    if (best_result.is_valid()) {
      algorithm_config.set_algorithm(best_result.algorithm());
    }
    if (best_result_no_scratch.is_valid()) {
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
    }
    AutoTuneConvBiasActivation::GetInstance()->Insert(conv_parameters,
                                                      algorithm_config);
  }

  CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
  bool cudnn_launch_status =
      stream
          ->ThenConvolveWithAlgorithm(
              input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
              bias_ptr, cudnn_activation_mode, output_desc, &output_ptr,
              &scratch_allocator, algorithm_config,
              /*output_profile_result=*/nullptr)
          .ok();

  if (!cudnn_launch_status) {
    ctx->SetStatus(errors::Internal(
        "cuDNN launch failure : input shape(", input.shape().DebugString(),
        ") filter shape(", filter.shape().DebugString(), ")"));
  }

  // Convert the output tensor back from NCHW to NHWC.
  if (data_format == FORMAT_NHWC) {
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
        output->tensor<T, 4>());
  } else {
    *output = transformed_output;
  }
}

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(Name("FusedConv2DBiasActivation")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        FusedConv2DBiasActivationOp<GPUDevice, float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
