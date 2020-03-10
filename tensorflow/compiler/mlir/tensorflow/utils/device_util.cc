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
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

constexpr char kDevicesAttr[] = "tf.devices";

namespace {

using DeviceNames = llvm::SmallVectorImpl<DeviceNameUtils::ParsedName>;

// Parse GPU compute capability from physical device description. If compute
// capability is not found in device description, return an empty dictionary
// attribute.
mlir::DictionaryAttr ParseGpuDeviceMetadata(const Device& device,
                                            mlir::Builder* builder) {
  // Parse GPU device compute capability from physical device description.
  static auto* r = new llvm::Regex("compute capability: ([0-9]+)\\.([0-9]+)");

  llvm::SmallVector<llvm::StringRef, 3> cc;
  if (r->match(device.attributes().physical_device_desc(), &cc)) {
    return mlir::TF::GpuDeviceMetadata::get(
        builder->getI32IntegerAttr(std::stoi(cc[1].str())),
        builder->getI32IntegerAttr(std::stoi(cc[2].str())),
        builder->getContext());
  }

  return builder->getDictionaryAttr({});
}

// Get device names from an array of string attributes.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::ArrayAttr array_attr,
                                     DeviceNames* devices) {
  devices->resize(array_attr.size());

  for (auto& kv : llvm::enumerate(array_attr)) {
    const int idx = kv.index();

    auto string_attr = kv.value().dyn_cast<mlir::StringAttr>();
    if (!string_attr)
      return op->emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, not a string", kDevicesAttr, idx));

    if (!DeviceNameUtils::ParseFullName(string_attr.getValue().str(),
                                        &(*devices)[idx]))
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, string_attr.getValue()));
  }

  return mlir::success();
}

// Get device names from a metadata dictionary.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::DictionaryAttr dict_attr,
                                     DeviceNames* devices) {
  devices->resize(dict_attr.size());

  // Parse device names and metadata from dictionary attribute.
  for (auto& kv : llvm::enumerate(dict_attr)) {
    const mlir::Identifier name = kv.value().first;

    if (!DeviceNameUtils::ParseFullName(name.str(), &(*devices)[kv.index()]))
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, name.strref()));
  }

  return mlir::success();
}

}  // namespace

void AddDevicesToOp(mlir::Operation* op, const DeviceSet* device_set) {
  if (!device_set) return;

  mlir::MLIRContext* ctx = op->getContext();
  mlir::Builder builder(ctx);

  // Collect devices with attached metadata.
  llvm::SmallVector<mlir::NamedAttribute, 8> devices;
  devices.reserve(device_set->devices().size());

  // For device that do not have any metadata, or if we failed to parse metadata
  // from the DeviceSet, we add empty dictionary to the `tf.devices` attribute.
  for (Device* device : device_set->devices()) {
    string name = DeviceNameUtils::ParsedNameToString(device->parsed_name());

    if (device->device_type() == DEVICE_GPU) {
      auto metadata = ParseGpuDeviceMetadata(*device, &builder);
      devices.push_back(builder.getNamedAttr(name, metadata));
    } else {
      auto metadata = builder.getDictionaryAttr({});
      devices.push_back(builder.getNamedAttr(name, metadata));
    }
  }

  op->setAttr(kDevicesAttr, builder.getDictionaryAttr(devices));
}

mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     DeviceNames* devices) {
  auto devices_attr = op->getAttr(kDevicesAttr);
  if (!devices_attr) return mlir::success();

  if (auto array_attr = devices_attr.dyn_cast<mlir::ArrayAttr>()) {
    return GetDevicesFromOp(op, array_attr, devices);

  } else if (auto dict_attr = devices_attr.dyn_cast<mlir::DictionaryAttr>()) {
    return GetDevicesFromOp(op, dict_attr, devices);
  }

  return op->emitOpError(
      llvm::formatv("unsupported '{0}' attribute", kDevicesAttr));
}

llvm::Optional<mlir::TF::GpuDeviceMetadata> GetGpuDeviceMetadata(
    mlir::Operation* op, const DeviceNameUtils::ParsedName& device) {
  auto metadata = op->getAttrOfType<mlir::DictionaryAttr>(kDevicesAttr);
  if (!metadata) return llvm::None;

  auto device_attr = metadata.get(DeviceNameUtils::ParsedNameToString(device));
  if (!device_attr) return llvm::None;

  return device_attr.dyn_cast<mlir::TF::GpuDeviceMetadata>();
}

}  // namespace tensorflow
