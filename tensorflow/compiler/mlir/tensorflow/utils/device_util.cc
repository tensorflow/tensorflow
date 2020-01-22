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

#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

constexpr char kDevicesAttr[] = "tf.devices";

void AddDevicesToOp(mlir::Operation* op, const DeviceSet* device_set) {
  if (!device_set) return;

  // Collect devices as strings in TensorFlow device name form.
  llvm::SmallVector<std::string, 8> devices;
  devices.reserve(device_set->devices().size());
  for (Device* device : device_set->devices())
    devices.push_back(
        DeviceNameUtils::ParsedNameToString(device->parsed_name()));

  llvm::SmallVector<llvm::StringRef, 8> device_refs(devices.begin(),
                                                    devices.end());
  mlir::Builder builder(op->getContext());
  op->setAttr(kDevicesAttr, builder.getStrArrayAttr(device_refs));
}

mlir::LogicalResult GetDevicesFromOp(
    mlir::Operation* op,
    llvm::SmallVectorImpl<DeviceNameUtils::ParsedName>* devices) {
  auto devices_attr = op->getAttr(kDevicesAttr);
  if (!devices_attr) return mlir::success();

  auto array_attr = devices_attr.dyn_cast<mlir::ArrayAttr>();
  if (!array_attr)
    return op->emitOpError(
        llvm::formatv("bad '{0}' attribute, not an array", kDevicesAttr));

  devices->resize(array_attr.size());
  for (auto attr_and_idx : llvm::enumerate(array_attr)) {
    const int idx = attr_and_idx.index();
    auto string_attr = attr_and_idx.value().dyn_cast<mlir::StringAttr>();
    if (!string_attr)
      return op->emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, not a string", kDevicesAttr, idx));

    if (!DeviceNameUtils::ParseFullName(string_attr.getValue().str(),
                                        &(*devices)[idx]))
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute at index {1} with value '{2}', "
                        "not a valid device",
                        kDevicesAttr, idx, string_attr.getValue()));
  }

  return mlir::success();
}

}  // namespace tensorflow
