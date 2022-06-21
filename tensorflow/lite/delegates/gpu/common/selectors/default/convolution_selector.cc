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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_constants.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal_simd.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {

std::unique_ptr<GPUOperation> SelectConvolutionAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  if (IsConvConstantsSupported(gpu_info, op_def, attr)) {
    GPUOperation conv = CreateConvConstants(gpu_info, op_def, attr);
    return std::make_unique<GPUOperation>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionWinogradAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  ConvPowerVR conv =
      CreateConvPowerVRWino4x4To6x6(gpu_info, op_def, attr, &dst_shape);
  return std::make_unique<ConvPowerVR>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConvolutionDynamicWeightsAdreno(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc) {
  ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
      gpu_info, op_def, attr, weights_shape, &dst_shape);
  *weights_desc = conv.GetWeightsDescription();
  return std::make_unique<ConvPowerVR>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConvolutionNVidia(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (IsConvConstantsSupported(gpu_info, op_def, attr)) {
    GPUOperation conv = CreateConvConstants(gpu_info, op_def, attr);
    return std::make_unique<GPUOperation>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionPowerVR(
    const Convolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, const BHWC& dst_shape) {
  ConvPowerVR conv = CreateConvPowerVR(gpu_info, op_def, attr, &dst_shape);
  return std::make_unique<ConvPowerVR>(std::move(conv));
}

std::unique_ptr<GPUOperation> SelectConvolutionMali(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER &&
      IsConvBuffer1x1Supported(op_def, attr)) {
    ConvBuffer1x1 conv =
        CreateConvBuffer1x1(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionWinogradMali(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    ConvBuffer1x1 conv =
        CreateConvBuffer1x1Wino4x4To6x6(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv =
        CreateConvPowerVRWino4x4To6x6(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionDynamicWeightsMali(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER &&
      IsConvBuffer1x1Supported(op_def, weights_shape, attr)) {
    ConvBuffer1x1 conv = CreateConvBuffer1x1DynamicWeights(
        gpu_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
        gpu_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionMetal(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def) {
  if (IsConvolutionMetalSimdSupported(gpu_info, op_def, attr) &&
      op_def.precision == CalculationsPrecision::F32 && gpu_info.IsApple() &&
      gpu_info.apple_info.IsSIMDMatMulFp32Perf2x() &&
      IsGoodTaskSizeForAppleConvSimd(dst_shape, gpu_info)) {
    ConvolutionMetalSimd conv =
        CreateConvolutionMetalSimd(op_def, dst_shape, attr, gpu_info);
    return std::make_unique<ConvolutionMetalSimd>(std::move(conv));
  } else if (IsConvolutionMetalSupported(op_def)) {
    ConvolutionMetal conv =
        CreateConvolutionMetal(op_def, dst_shape, attr, gpu_info);
    return std::make_unique<ConvolutionMetal>(std::move(conv));
  } else {
    ConvPowerVR conv = CreateConvPowerVR(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

}  // namespace

std::unique_ptr<GPUOperation> SelectConvolution(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  if (gpu_info.IsApiMetal()) {
    return SelectConvolutionMetal(attr, dst_shape, gpu_info, op_def);
  } else if (gpu_info.IsAdreno()) {
    return SelectConvolutionAdreno(attr, dst_shape, gpu_info, op_def, hints);
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsIntel() ||
             gpu_info.IsApple()) {
    return SelectConvolutionPowerVR(attr, gpu_info, op_def, dst_shape);
  } else if (gpu_info.IsNvidia()) {
    return SelectConvolutionNVidia(attr, dst_shape, gpu_info, op_def);
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionMali(attr, dst_shape, gpu_info, op_def);
  } else {
    return SelectConvolutionAdreno(attr, dst_shape, gpu_info, op_def, hints);
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionForWinograd(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ModelHints hints) {
  if (gpu_info.IsApiMetal() && IsConvolutionMetalSupported(op_def)) {
    ConvolutionMetal conv =
        CreateConvolutionMetalWino4x4To6x6(op_def, dst_shape, attr, gpu_info);
    return std::make_unique<ConvolutionMetal>(std::move(conv));
  } else if (gpu_info.IsAdreno()) {
    return SelectConvolutionWinogradAdreno(attr, dst_shape, gpu_info, op_def,
                                           hints);
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsNvidia() ||
             gpu_info.IsIntel() || gpu_info.IsApple()) {
    ConvPowerVR conv =
        CreateConvPowerVRWino4x4To6x6(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvPowerVR>(std::move(conv));
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionWinogradMali(attr, dst_shape, gpu_info, op_def);
  } else {
    return SelectConvolutionWinogradAdreno(attr, dst_shape, gpu_info, op_def,
                                           hints);
  }
}

std::unique_ptr<GPUOperation> SelectConvolutionWithDynamicWeights(
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC& dst_shape, const GpuInfo& gpu_info,
    const OperationDef& op_def, ModelHints hints,
    WeightsDescription* weights_desc) {
  if (gpu_info.IsApiMetal() && IsConvolutionMetalSupported(op_def)) {
    Convolution2DAttributes attr_copy = attr;
    attr_copy.weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                   weights_shape.w, weights_shape.c);
    ConvolutionMetal conv =
        CreateConvolutionMetal(op_def, dst_shape, attr_copy, gpu_info);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvolutionMetal>(std::move(conv));
  } else if (gpu_info.IsAdreno()) {
    return SelectConvolutionDynamicWeightsAdreno(attr, weights_shape, dst_shape,
                                                 gpu_info, op_def, hints,
                                                 weights_desc);
  } else if (gpu_info.IsMali()) {
    return SelectConvolutionDynamicWeightsMali(attr, weights_shape, dst_shape,
                                               gpu_info, op_def, hints,
                                               weights_desc);
  } else {
    ConvPowerVR conv = CreateConvPowerVRDynamicWeights(
        gpu_info, op_def, attr, weights_shape, &dst_shape);
    *weights_desc = conv.GetWeightsDescription();
    return std::make_unique<ConvPowerVR>(std::move(conv));
  }
}

std::unique_ptr<GPUOperation> SelectConverterToConvWeights(
    const WeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints) {
  ConverterToConvWeights converter =
      ConverterToConvWeights(op_def, weights_desc);
  return std::make_unique<ConverterToConvWeights>(std::move(converter));
}

}  // namespace gpu
}  // namespace tflite
