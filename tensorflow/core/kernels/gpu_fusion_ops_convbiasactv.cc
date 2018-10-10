/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <string.h>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/gpu_fusion_ops.h"

#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ROCmFusionKernelConvolutionBiasActivation : public OpKernel {
 public:
  explicit ROCmFusionKernelConvolutionBiasActivation(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_type_));

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    string filter_format_str("HWIO");
    OP_REQUIRES(ctx, FilterFormatFromString(filter_format_str, &filter_format_),
                errors::InvalidArgument("Invalid filter format"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));

    string activation_mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(ctx, GetActivationModeFromString(activation_mode_str,
                                                    &activation_mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& conv_input = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    const int32 batch_size = GetTensorDim(conv_input, data_format_, 'N');
    int32 input_rows = GetTensorDim(conv_input, data_format_, 'H');
    int32 input_cols = GetTensorDim(conv_input, data_format_, 'W');
    const int32 input_channels = GetTensorDim(conv_input, data_format_, 'C');

    const int32 filter_rows = GetFilterDim(filter, filter_format_, 'H');
    const int32 filter_cols = GetFilterDim(filter, filter_format_, 'W');
    const int32 output_channels = GetFilterDim(filter, filter_format_, 'O');

    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'H');

    int64 output_rows = 0, padding_left = 0, padding_right = 0;
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeVerboseV2(
                 input_rows, filter_rows, dilation_rows, stride_rows,
                 padding_type_, &output_rows, &padding_left, &padding_right));
    int64 padding_rows = padding_left + padding_right;

    int64 output_cols = 0, padding_top = 0, padding_bottom = 0;
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeVerboseV2(
                 input_cols, filter_cols, dilation_cols, stride_cols,
                 padding_type_, &output_cols, &padding_top, &padding_bottom));
    int64 padding_cols = padding_top + padding_bottom;

    TensorShape output_shape = ShapeFromFormat(
        data_format_, batch_size, output_rows, output_cols, output_channels);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0 /*output index*/, output_shape, &output));

    if (output_shape.num_elements() != 0) {
      Tensor fusion_input = conv_input;
      Tensor fusion_filter = filter;
      Tensor fusion_bias = bias;
      Tensor fusion_output = *output;

      Tensor transformed_input, transformed_filter, transformed_output;

      // if the padding type is SAME, we need to pad the input first
      if (padding_type_ == Padding::SAME) {
        // if either the padding_rows and padding_cols are odd, we need to a
        // zero padding row and/or col to the input
        const bool rows_odd = (padding_rows % 2);
        const bool cols_odd = (padding_cols % 2);
        if (rows_odd || cols_odd) {
          int64 new_input_rows = input_rows + rows_odd;
          int64 new_input_cols = input_cols + cols_odd;

          // allocate a temporary tensor to store the padded input
          Tensor padded_input;
          TensorShape new_input_shape =
              ShapeFromFormat(data_format_, batch_size, new_input_rows,
                              new_input_cols, input_channels);
          OP_REQUIRES_OK(
              ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, new_input_shape,
                                      &padded_input));

          // add padding to the input
          functor::PadInput<GPUDevice, T, int, 4>()(
              ctx->eigen_device<GPUDevice>(),
              To32Bit(const_cast<const Tensor&>(fusion_input).tensor<T, 4>()),
              {{0, 0}}, {{rows_odd, cols_odd}},
              To32Bit(padded_input.tensor<T, 4>()), data_format_);

          fusion_input = padded_input;

          input_rows = new_input_rows;
          input_cols = new_input_cols;
        }
      }

      // if the data format is NHWC, we need to
      // 1. convert the input tensor to NCHW format
      // 2, allocate a temporary tensor to hold the fusion op output (which will
      // be in NCHW format)
      if (data_format_ == FORMAT_NHWC) {
        // allocate a temporary tensor to store the NCHW input
        Tensor transformed_input;
        TensorShape nchw_shape_input = ShapeFromFormat(
            FORMAT_NCHW, batch_size, input_rows, input_cols, input_channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input,
                                    &transformed_input));

        // convert the input tensor to NCHW format for the GPU
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());

        fusion_input = transformed_input;

        // allocate a temporary tensor to store the NCHW output
        Tensor transformed_output;
        TensorShape nchw_shape_output = ShapeFromFormat(
            FORMAT_NCHW, batch_size, output_rows, output_cols, output_channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_output,
                                    &transformed_output));

        fusion_output = transformed_output;
      }

      // if the filter format in HWIO, we need to convert the filter tensor to
      // OIHW format
      if (filter_format_ == FORMAT_HWIO) {
        // allocate a temporary tensor to store the OIHW filter
        Tensor transformed_filter;
        TensorShape oihw_shape_filter =
            ShapeFromFilterFormat(FORMAT_OIHW, filter.shape(), FORMAT_HWIO);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, oihw_shape_filter,
                                    &transformed_filter));

        // convert the input tensor to OIHW format for the GPU
        functor::TransformFilter<GPUDevice, T, int, 4>()(
            ctx->eigen_device<GPUDevice>(), FORMAT_OIHW,
            To32Bit(const_cast<const Tensor&>(fusion_filter).tensor<T, 4>()),
            To32Bit(transformed_filter.tensor<T, 4>()));

        fusion_filter = transformed_filter;
      }

      se::dnn::BatchDescriptor conv_input_desc;
      conv_input_desc.set_count(batch_size)
          .set_feature_map_count(input_channels)
          .set_height(input_rows)
          .set_width(input_cols)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::FilterDescriptor filter_desc;
      filter_desc.set_input_filter_height(filter_rows)
          .set_input_filter_width(filter_cols)
          .set_input_feature_map_count(input_channels)
          .set_output_feature_map_count(output_channels)
          .set_layout(se::dnn::FilterLayout::kOutputInputYX);

      se::dnn::BatchDescriptor bias_desc;
      bias_desc.set_count(1)
          .set_height(1)
          .set_width(1)
          .set_feature_map_count(output_channels)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::BatchDescriptor output_desc;
      output_desc.set_count(batch_size)
          .set_height(output_rows)
          .set_width(output_cols)
          .set_feature_map_count(output_channels)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::ConvolutionDescriptor conv_desc;
      conv_desc.set_vertical_filter_stride(stride_rows)
          .set_horizontal_filter_stride(stride_cols)
          .set_zero_padding_height(padding_rows / 2)
          .set_zero_padding_width(padding_cols / 2);

      auto conv_input_data =
          AsDeviceMemory(fusion_input.template flat<T>().data(),
                         fusion_input.template flat<T>().size());
      auto filter_data =
          AsDeviceMemory(fusion_filter.template flat<T>().data(),
                         fusion_filter.template flat<T>().size());
      auto bias_data = AsDeviceMemory(fusion_bias.template flat<T>().data(),
                                      fusion_bias.template flat<T>().size());
      auto dnn_activation_mode = GetDnnActivationMode(activation_mode_);

      auto output_data =
          AsDeviceMemory(fusion_output.template flat<T>().data(),
                         fusion_output.template flat<T>().size());

      auto* stream = ctx->op_device_context()->stream();

      bool miopen_status =
          stream
              ->ThenFusedConvolutionBiasActivation(
                  conv_input_desc, conv_input_data, filter_desc, filter_data,
                  conv_desc, bias_desc, bias_data, dnn_activation_mode,
                  output_desc, &output_data)
              .ok();

      if (!miopen_status) {
        ctx->SetStatus(errors::Internal("MIOpen CBA FusionOp launch Failure"));
      }

      // if the data format is NHWC, we need to convert the fusion op output
      // back to MHWC format
      if (data_format_ == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_output).tensor<T, 4>(),
            output->tensor<T, 4>());
      }
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_type_;
  TensorFormat data_format_;
  FilterTensorFormat filter_format_;
  std::vector<int32> dilations_;
  ActivationMode activation_mode_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedConvolutionBiasActivation")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    ROCmFusionKernelConvolutionBiasActivation<GPUDevice, float>);

// Forward declarations of the functor specializations for GPU used above.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);

#undef DECLARE_GPU_SPEC
}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
