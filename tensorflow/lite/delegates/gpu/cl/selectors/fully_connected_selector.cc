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

#include "tensorflow/lite/delegates/gpu/cl/selectors/fully_connected_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

std::unique_ptr<GPUOperation> SelectFullyConnectedGeneric(
    const FullyConnectedAttributes& attr, const DeviceInfo& device_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    ConvTexture conv = CreateConvTexture(device_info, op_def, attr);
    return absl::make_unique<ConvTexture>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(device_info, op_def, attr);
    return absl::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedAdreno(
    const FullyConnectedAttributes& attr, const DeviceInfo& device_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    ConvTexture conv = CreateConvTexture(device_info, op_def, attr);
    return absl::make_unique<ConvTexture>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(device_info, op_def, attr);
    return absl::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedPowerVR(
    const FullyConnectedAttributes& attr, const DeviceInfo& device_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    ConvPowerVR conv = CreateConvPowerVR(device_info, op_def, attr);
    return absl::make_unique<ConvPowerVR>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(device_info, op_def, attr);
    return absl::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnectedMali(
    const FullyConnectedAttributes& attr, const DeviceInfo& device_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
      ConvBuffer1x1 conv = CreateConvBuffer1x1(device_info, op_def, attr);
      return absl::make_unique<ConvBuffer1x1>(std::move(conv));
    } else {
      ConvTexture conv = CreateConvTexture(device_info, op_def, attr);
      return absl::make_unique<ConvTexture>(std::move(conv));
    }
  } else {
    FullyConnected fc = CreateFullyConnected(device_info, op_def, attr);
    return absl::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedAttributes& attr, const DeviceInfo& device_info,
    const OperationDef& op_def, int batch_size) {
  if (device_info.IsAdreno()) {
    return SelectFullyConnectedAdreno(attr, device_info, op_def, batch_size);
  } else if (device_info.IsPowerVR() || device_info.IsAMD() ||
             device_info.IsNvidia() || device_info.IsIntel()) {
    return SelectFullyConnectedPowerVR(attr, device_info, op_def, batch_size);
  } else if (device_info.IsMali()) {
    return SelectFullyConnectedMali(attr, device_info, op_def, batch_size);
  } else {
    return SelectFullyConnectedGeneric(attr, device_info, op_def, batch_size);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
