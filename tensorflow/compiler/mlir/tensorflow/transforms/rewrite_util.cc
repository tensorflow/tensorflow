/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.h"

#include <optional>
#include <string>

#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TF {

namespace {

const char kDeviceAttr[] = "device";
const char kDeviceGpu[] = "GPU";

std::optional<std::string> GetOpDevice(mlir::Operation *op) {
  mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
  if (!device || device.getValue().empty()) {
    return std::nullopt;
  }
  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(device.str(), &parsed_name)) {
    return std::nullopt;
  }
  if (!parsed_name.has_type) {
    return std::nullopt;
  }
  return parsed_name.type;
}

}  // namespace

bool IsOnGpuDevice(mlir::Operation *op) {
  std::optional<std::string> device = GetOpDevice(op);
  if (!device) return false;
  return *device == kDeviceGpu;
}

}  // namespace TF
}  // namespace mlir
