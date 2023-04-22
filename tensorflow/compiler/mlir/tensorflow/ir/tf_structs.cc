/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.cc.inc"

namespace mlir {
namespace TF {

void RuntimeDevices::AddDevice(const ParsedName& device) {
  device_names_.push_back(device);
}

void RuntimeDevices::AddGpuDevice(const ParsedName& device,
                                  const GpuDeviceMetadata& metadata) {
  device_names_.push_back(device);
  gpu_metadata_.insert({DeviceNameUtils::ParsedNameToString(device), metadata});
}

llvm::Optional<GpuDeviceMetadata> RuntimeDevices::GetGpuDeviceMetadata(
    const ParsedName& device) const {
  auto it = gpu_metadata_.find(DeviceNameUtils::ParsedNameToString(device));
  if (it != gpu_metadata_.end()) {
    return it->second;
  } else {
    return llvm::None;
  }
}

}  // namespace TF
}  // namespace mlir
