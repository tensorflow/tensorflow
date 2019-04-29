/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/backends/acl/depthwise_conv2d.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/backends/acl/utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

void ACLDepthwiseConv2dBackendKernel::init(TfLiteContext* context,
                                           const char* buffer, size_t length) {
  // no-op for builtin ops
}

void ACLDepthwiseConv2dBackendKernel::free(TfLiteContext* context) {
  // no-op for builtin ops
}

TfLiteStatus ACLDepthwiseConv2dBackendKernel::prepare(TfLiteContext* context,
                                                      TfLiteNode* node) {
  _is_configured = false;

  const auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const bool has_bias = node->inputs->size == 3;

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* bias =
      has_bias ? &context->tensors[node->inputs->data[2]] : nullptr;

  // Data type
  const auto io_data_type = map_datatype_to_acl(input->type);
  const auto filter_data_type = map_datatype_to_acl(filter->type);
  const auto bias_data_type = bias != nullptr ? map_datatype_to_acl(bias->type)
                                              : arm_compute::DataType::UNKNOWN;
  ACL_RETURN_ERROR_ON(io_data_type != arm_compute::DataType::F32);
  ACL_RETURN_ERROR_ON(io_data_type != filter_data_type);

  // Tensor shapes
  const auto input_shape = map_shape_to_acl(*input->dims);
  const auto filter_shape = map_shape_to_acl(*filter->dims);
  const auto bias_shape = arm_compute::TensorShape(filter_shape[0]);
  const auto output_shape = map_shape_to_acl(*output->dims);

  // Quantization info
  const auto input_qinfo = map_quantization_info_to_acl(input->quantization);
  const auto filter_qinfo = map_quantization_info_to_acl(filter->quantization);
  const auto output_qinfo = map_quantization_info_to_acl(output->quantization);
  if (io_data_type != arm_compute::DataType::F32) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const int number_channel = affine_quantization->scale->size;
    // Only asymmetric is supported
    ACL_RETURN_ERROR_ON(number_channel != 1);
    const double effective_output_scale =
        static_cast<double>(input_qinfo.scale) * filter_qinfo.scale /
        static_cast<double>(output_qinfo.scale);
    ACL_RETURN_ERROR_ON(effective_output_scale > 1.f);
  }

  // Construct TensorInfo and set data layout to NHWC
  arm_compute::TensorInfo input_info(input_shape, 1, io_data_type, input_qinfo);
  arm_compute::TensorInfo filter_info(filter_shape, 1, filter_data_type,
                                      filter_qinfo);
  arm_compute::TensorInfo bias_info(bias_shape, 1, bias_data_type);
  arm_compute::TensorInfo output_info(output_shape, 1, io_data_type,
                                      output_qinfo);
  input_info.set_data_layout(arm_compute::DataLayout::NHWC);
  filter_info.set_data_layout(arm_compute::DataLayout::NHWC);
  output_info.set_data_layout(arm_compute::DataLayout::NHWC);

  // Calculate padding and create PadStrideInfo
  const arm_compute::Size2D strides(params->stride_width,
                                    params->stride_height);
  const TfLitePadding padding_type = params->padding;
  const auto conv_info = map_pad_stride_info_to_acl(strides, padding_type,
                                                    input_shape, filter_shape);

  // Extract activation and fuse with convolution
  const auto activation_info = map_activation_info_to_acl(params->activation);
  ACL_RETURN_ERROR_ON(
      activation_info.enabled() &&
      (arm_compute::ActivationLayerInfo::ActivationFunction::RELU !=
           activation_info.activation() &&
       arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU !=
           activation_info.activation()));

  // Extract dilation
  const arm_compute::Size2D dilation(params->dilation_width_factor,
                                     params->dilation_height_factor);
  ACL_RETURN_ERROR_ON(dilation != arm_compute::Size2D(1U, 1U));

  // Depth multiplier
  const int depth_multiplier = params->depth_multiplier;
  ACL_RETURN_ERROR_ON(depth_multiplier != 1);

  // Validate if ACL function can run
  arm_compute::Status s = arm_compute::NEDepthwiseConvolutionLayer3x3::validate(
      &input_info.clone()->set_is_resizable(false),
      &filter_info.clone()->set_is_resizable(false),
      has_bias ? &bias_info.clone()->set_is_resizable(false) : nullptr,
      &output_info.clone()->set_is_resizable(false), conv_info,
      depth_multiplier, activation_info, dilation);

  if (bool(s)) {
    // Initialize tensors
    _input.allocator()->init(input_info);
    _filter.allocator()->init(filter_info);
    _bias.allocator()->init(bias_info);
    _output.allocator()->init(output_info);

    // Configure function
    _conv_func.configure(&_input, &_filter, has_bias ? &_bias : nullptr,
                         &_output, conv_info, depth_multiplier, activation_info,
                         dilation);
  }
  _is_configured = bool(s);

  return _is_configured ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ACLDepthwiseConv2dBackendKernel::invoke(TfLiteContext* context,
                                                     TfLiteNode* node) {
  if (_is_configured) {
    bool has_bias = node->inputs->size == 3;

    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
    TfLiteTensor* bias =
        has_bias ? &context->tensors[node->inputs->data[2]] : nullptr;

    // Setup tensor memory
    _input.allocator()->import_memory(reinterpret_cast<void*>(input->data.raw));
    _filter.allocator()->import_memory(
        reinterpret_cast<void*>(filter->data.raw));
    if (has_bias) {
      _bias.allocator()->import_memory(reinterpret_cast<void*>(bias->data.raw));
    }
    _output.allocator()->import_memory(
        reinterpret_cast<void*>(output->data.raw));

    // Run backend ACL function
    _conv_func.run();
  }
}
}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
