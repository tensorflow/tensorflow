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

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_generic.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectFullyConnectedGeneric(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedAdreno(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedPowerVR(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedMali(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    if (op_def.src_tensors[0].GetStorageType() == TensorStorageType::BUFFER) {
      ConvBuffer1x1 conv = CreateConvBuffer1x1(gpu_info, op_def, attr);
      return std::make_unique<ConvBuffer1x1>(std::move(conv));
    } else {
      BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
      ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
      return std::make_unique<ConvGeneric>(std::move(conv));
    }
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (gpu_info.IsApple()) {
    if (op_def.IsBatchSupported() && IsConvolutionMetalSupported(op_def)) {
      BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
      Convolution2DAttributes conv_attr;
      conv_attr.padding.prepended = HW(0, 0);
      conv_attr.padding.appended = HW(0, 0);
      conv_attr.strides = HW(1, 1);
      conv_attr.dilations = HW(1, 1);
      conv_attr.weights = attr.weights;
      conv_attr.bias = attr.bias;
      ConvolutionMetal conv =
          CreateConvolutionMetal(op_def, dst_shape, conv_attr, gpu_info);
      return std::make_unique<ConvolutionMetal>(std::move(conv));
    } else {
      FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
      return std::make_unique<FullyConnected>(std::move(fc));
    }
  } else if (gpu_info.IsAdreno()) {
    return SelectFullyConnectedAdreno(attr, gpu_info, op_def, batch_size);
  } else if (gpu_info.IsPowerVR() || gpu_info.IsAMD() || gpu_info.IsNvidia() ||
             gpu_info.IsIntel() || gpu_info.IsApple()) {
    return SelectFullyConnectedPowerVR(attr, gpu_info, op_def, batch_size);
  } else if (gpu_info.IsMali()) {
    return SelectFullyConnectedMali(attr, gpu_info, op_def, batch_size);
  } else {
    return SelectFullyConnectedGeneric(attr, gpu_info, op_def, batch_size);
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedInt8Attributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
  return std::make_unique<FullyConnected>(std::move(fc));
}

}  // namespace gpu
}  // namespace tflite
