/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_device_helper.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TF {
namespace {
using DeviceNameUtils = ::tensorflow::DeviceNameUtils;
using ParsedName = ::tensorflow::DeviceNameUtils::ParsedName;

bool IsGpuDevice(const DeviceNameUtils::ParsedName &device) {
  return device.type == ::tensorflow::DEVICE_GPU;
}

}  // namespace

// Returns true if at least one GPU device is available at runtime.
bool CanUseGpuDevice(const RuntimeDevices &devices) {
  return llvm::any_of(devices.device_names(), IsGpuDevice);
}

// Returns true if all of the GPUs available at runtime support TensorCores
// (NVIDIA compute capability >= 7.0).
bool CanUseTensorCores(const RuntimeDevices &devices) {
  auto has_tensor_cores = [&](const DeviceNameUtils::ParsedName &device) {
    auto md = devices.GetGpuDeviceMetadata(device);
    return md ? md->cc_major().getInt() >= 7 : false;
  };
  return llvm::all_of(
      llvm::make_filter_range(devices.device_names(), IsGpuDevice),
      has_tensor_cores);
}

// Returns true if operation does not have explicit device placement that would
// prevent it from running on GPU device.
bool CanUseGpuDevice(Operation *op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  if (!device_attr || device_attr.getValue().empty()) return true;

  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_attr.getValue().str(), &device))
    return false;

  // We can't use GPU if operation explicitly placed on non-GPU device.
  return !device.has_type || device.type == ::tensorflow::DEVICE_GPU;
}

}  // namespace TF
}  // namespace mlir
