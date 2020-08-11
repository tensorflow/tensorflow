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

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

constexpr char kDevicesAttr[] = "tf.devices";

namespace {

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

// Get devices from an array of string attributes.
// TODO(ezhulenev): Update all tests to use dictionary attribute for
// `tf.devices` and remove this function.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::ArrayAttr array_attr,
                                     mlir::TF::RuntimeDevices* devices) {
  DeviceNameUtils::ParsedName device;

  for (auto& kv : llvm::enumerate(array_attr)) {
    const int idx = kv.index();

    auto string_attr = kv.value().dyn_cast<mlir::StringAttr>();
    if (!string_attr)
      return op->emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1}, not a string", kDevicesAttr, idx));

    if (DeviceNameUtils::ParseFullName(string_attr.getValue().str(), &device)) {
      devices->AddDevice(device);
    } else {
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, string_attr.getValue()));
    }
  }

  return mlir::success();
}

// Get devices from a dictionary attribute.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::DictionaryAttr dict_attr,
                                     mlir::TF::RuntimeDevices* devices) {
  DeviceNameUtils::ParsedName device;

  // Parse device names and metadata from dictionary attribute.
  for (auto& kv : dict_attr) {
    const mlir::Identifier name = kv.first;
    const mlir::Attribute attr = kv.second;

    if (!DeviceNameUtils::ParseFullName(name.str(), &device))
      return op->emitOpError(
          llvm::formatv("bad '{0}' attribute, '{1}', not a valid device",
                        kDevicesAttr, name.strref()));

    if (auto gpu_metadata = attr.dyn_cast<mlir::TF::GpuDeviceMetadata>()) {
      devices->AddGpuDevice(device, gpu_metadata);
    } else {
      devices->AddDevice(device);
    }
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
                                     mlir::TF::RuntimeDevices* devices) {
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

mlir::LogicalResult GetDeviceOrdinalFromDeviceString(mlir::Location loc,
                                                     llvm::StringRef device,
                                                     int64_t* device_ordinal) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(
          absl::string_view(device.data(), device.size()), &parsed_name))
    return mlir::emitError(loc) << "invalid device '" << device << "'";

  if (!parsed_name.has_id)
    return mlir::emitError(loc) << "device '" << device << "' has no id";

  *device_ordinal = parsed_name.id;
  return mlir::success();
}

}  // namespace tensorflow
