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

#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_constants.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_weights_converter.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

absl::Status SelectConvolutionAdreno(const Convolution2DAttributes& attr,
                                     const BHWC& dst_shape,
                                     const DeviceInfo& device_info,
                                     const OperationDef& op_def,
                                     ModelHints hints,
                                     std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(device_info, op_def, attr)) {
    ConvConstants conv = CreateConvConstants(device_info, op_def, attr);
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvTexture conv = CreateConvTexture(device_info, op_def, attr);
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionWinogradAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const DeviceInfo& device_info, const OperationDef& op_def, ModelHints hints,
    std::unique_ptr<GPUOperation>* ptr) {
  ConvTexture conv = CreateConvTextureWino4x4To6x6(device_info, op_def, attr);
  *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  return absl::OkStatus();
}

absl::Status SelectConvolutionDynamicWeightsAdreno(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const DeviceInfo& device_info,
    const OperationDef& op_def, ModelHints hints,
    std::unique_ptr<GPUOperation>* ptr, ConvWeightsDescription* weights_desc) {
  ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
      device_info, op_def, attr, weights_shape, &dst_shape);
  *weights_desc = conv.GetConvWeightsDescription();
  *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  return absl::OkStatus();
}

absl::Status SelectConvolutionNVidia(const Convolution2DAttributes& attr,
                                     const BHWC& dst_shape,
                                     const DeviceInfo& device_info,
                                     const OperationDef& op_def,
                                     std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(device_info, op_def, attr)) {
    ConvConstants conv = CreateConvConstants(device_info, op_def, attr);
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionPowerVR(const Convolution2DAttributes& attr,
                                      const DeviceInfo& device_info,
                                      const OperationDef& op_def,
                                      std::unique_ptr<GPUOperation>* ptr) {
  ConvPowerVR conv = CreateConvPowerVR(device_info, op_def, attr);
  *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  return absl::OkStatus();
}

absl::Status SelectConvolutionMali(const Convolution2DAttributes& attr,
                                   const BHWC& dst_shape,
                                   const DeviceInfo& device_info,
                                   const OperationDef& op_def,
                                   std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER &&
      IsConvBuffer1x1Supported(op_def, attr)) {
    ConvBuffer1x1 conv =
        CreateConvBuffer1x1(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionWinogradMali(const Convolution2DAttributes& attr,
                                           const BHWC& dst_shape,
                                           const DeviceInfo& device_info,
                                           const OperationDef& op_def,
                                           std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    ConvBuffer1x1 conv =
        CreateConvBuffer1x1Wino4x4To6x6(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv =
        CreateConvPowerVRWino4x4To6x6(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }

  return absl::OkStatus();
}

absl::Status SelectConvolutionDynamicWeightsMali(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const DeviceInfo& device_info,
    const OperationDef& op_def, ModelHints hints,
    std::unique_ptr<GPUOperation>* ptr, ConvWeightsDescription* weights_desc) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER &&
      IsConvBuffer1x1Supported(op_def, weights_shape, attr)) {
    ConvBuffer1x1 conv = CreateConvBuffer1x1DynamicWeights(
        device_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetConvWeightsDescription();
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
        device_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetConvWeightsDescription();
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SelectConvolution(const Convolution2DAttributes& attr,
                               const BHWC& dst_shape,
                               const DeviceInfo& device_info,
                               const OperationDef& op_def, ModelHints hints,
                               std::unique_ptr<GPUOperation>* ptr) {
  if (device_info.IsAdreno()) {
    return SelectConvolutionAdreno(attr, dst_shape, device_info, op_def, hints,
                                   ptr);
  } else if (device_info.IsPowerVR() || device_info.IsAMD() ||
             device_info.IsIntel()) {
    return SelectConvolutionPowerVR(attr, device_info, op_def, ptr);
  } else if (device_info.IsNvidia()) {
    return SelectConvolutionNVidia(attr, dst_shape, device_info, op_def, ptr);
  } else if (device_info.IsMali()) {
    return SelectConvolutionMali(attr, dst_shape, device_info, op_def, ptr);
  } else {
    return SelectConvolutionAdreno(attr, dst_shape, device_info, op_def, hints,
                                   ptr);
  }
}

absl::Status SelectConvolutionForWinograd(const Convolution2DAttributes& attr,
                                          const BHWC& dst_shape,
                                          const DeviceInfo& device_info,
                                          const OperationDef& op_def,
                                          ModelHints hints,
                                          std::unique_ptr<GPUOperation>* ptr) {
  if (device_info.IsAdreno()) {
    return SelectConvolutionWinogradAdreno(attr, dst_shape, device_info, op_def,
                                           hints, ptr);
  } else if (device_info.IsPowerVR() || device_info.IsAMD() ||
             device_info.IsNvidia() || device_info.IsIntel()) {
    ConvPowerVR conv =
        CreateConvPowerVRWino4x4To6x6(device_info, op_def, attr, &dst_shape);
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
    return absl::OkStatus();
  } else if (device_info.IsMali()) {
    return SelectConvolutionWinogradMali(attr, dst_shape, device_info, op_def,
                                         ptr);
  } else {
    return SelectConvolutionWinogradAdreno(attr, dst_shape, device_info, op_def,
                                           hints, ptr);
  }
}

absl::Status SelectConvolutionWithDynamicWeights(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const DeviceInfo& device_info,
    const OperationDef& op_def, ModelHints hints,
    std::unique_ptr<GPUOperation>* ptr, ConvWeightsDescription* weights_desc) {
  if (device_info.IsAdreno()) {
    return SelectConvolutionDynamicWeightsAdreno(attr, weights_shape, dst_shape,
                                                 device_info, op_def, hints,
                                                 ptr, weights_desc);
  } else if (device_info.IsMali()) {
    return SelectConvolutionDynamicWeightsMali(attr, weights_shape, dst_shape,
                                               device_info, op_def, hints, ptr,
                                               weights_desc);
  } else {
    ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
        device_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetConvWeightsDescription();
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
    return absl::OkStatus();
  }
}

absl::Status SelectConverterToConvWeights(
    const ConvWeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints, std::unique_ptr<GPUOperation>* ptr) {
  ConverterToConvWeights converter =
      ConverterToConvWeights(op_def, weights_desc);
  *ptr = absl::make_unique<ConverterToConvWeights>(std::move(converter));
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
