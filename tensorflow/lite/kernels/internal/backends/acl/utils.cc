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
#include "tensorflow/lite/kernels/internal/backends/acl/utils.h"

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

arm_compute::DataType map_datatype_to_acl(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return arm_compute::DataType::F32;
    case kTfLiteInt32:
      return arm_compute::DataType::S32;
    case kTfLiteUInt8:
      return arm_compute::DataType::QASYMM8;
    default:
      return arm_compute::DataType::UNKNOWN;
  }
}

arm_compute::TensorShape map_shape_to_acl(const TfLiteIntArray& dims) {
  arm_compute::TensorShape shape;

  const int num_dimensions = dims.size;
  for (int i = num_dimensions - 1; i >= 0; --i) {
    shape.set(num_dimensions - i - 1, dims.data[i]);
  }
  return shape;
}

arm_compute::ActivationLayerInfo map_activation_info_to_acl(
    TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActRelu:
      return arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    case kTfLiteActRelu1:
      return arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
          1.f, 0.f);
    case kTfLiteActRelu6:
      return arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
          6.f, 0.f);
    case kTfLiteActTanh:
      return arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f);
    case kTfLiteActSigmoid:
      return arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
    default:
      return arm_compute::ActivationLayerInfo();
  }
}

arm_compute::QuantizationInfo map_quantization_info_to_acl(
    const TfLiteQuantization& quantization) {
  arm_compute::QuantizationInfo qinfo;
  if (quantization.type == kTfLiteAffineQuantization) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(quantization.params);
    float scale = affine_quantization->scale->data[0];
    int zero_point = affine_quantization->zero_point->data[0];
    qinfo = arm_compute::QuantizationInfo(scale, zero_point);
  }

  return qinfo;
}

arm_compute::PadStrideInfo map_pad_stride_info_to_acl(
    const arm_compute::Size2D& strides, TfLitePadding padding,
    const arm_compute::TensorShape& input_shape,
    const arm_compute::TensorShape& weights_shape) {
  const int stride_w = strides.width;
  const int stride_h = strides.height;

  const int width = input_shape[1];
  const int height = input_shape[2];
  const int filter_width = weights_shape[1];
  const int filter_height = weights_shape[2];

  arm_compute::PadStrideInfo conv_info(stride_w, stride_h, 0, 0);
  if (padding == TfLitePadding::kTfLitePaddingSame) {
    const int out_width = std::ceil(float(width) / float(stride_w));
    const int out_height = std::ceil(float(height) / float(stride_h));
    const int pad_width = ((out_width - 1) * stride_w + filter_width - width);
    const int pad_height =
        ((out_height - 1) * stride_h + filter_height - height);
    const int same_pad_left = pad_width / 2;
    const int same_pad_top = pad_height / 2;
    const int same_pad_right = pad_width - same_pad_left;
    const int same_pad_bottom = pad_height - same_pad_top;

    conv_info = arm_compute::PadStrideInfo(
        stride_w, stride_h, same_pad_left, same_pad_right, same_pad_top,
        same_pad_bottom, arm_compute::DimensionRoundingType::FLOOR);
  }

  return conv_info;
}

}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
