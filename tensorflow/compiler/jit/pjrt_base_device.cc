/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/jit/pjrt_base_device.h"

namespace tensorflow {
namespace {

DeviceAttributes BuildPjRtBaseDeviceAttributes(const string& name_prefix,
                                               const string& device_name,
                                               int device_ordinal) {
  return Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", device_name, ":", device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      absl::StrCat("device: ", device_name, " device"));
}

}  // namespace

PjRtBaseDevice::PjRtBaseDevice(const SessionOptions& session_options,
                               const Options& options)
    : LocalDevice(session_options,
                  BuildPjRtBaseDeviceAttributes(options.device_name_prefix,
                                                options.device_name,
                                                options.device_ordinal)),
      metadata_(DeviceType(options.compilation_device_name),
                options.shape_determination_fns) {
  if (options.shape_determination_fns.empty()) {
    LOG(ERROR) << "shape_representation_fns must be non-empty.";
  }
  VLOG(1) << "Created PJRT base device " << options.compilation_device_name
          << " device_name: " << name();
}

/*static*/ absl::StatusOr<const PjRtBaseDevice::Metadata*>
PjRtBaseDevice::GetMetadataFromDevice(DeviceBase* device) {
  PjRtBaseDevice* pjrt_device =
      dynamic_cast<PjRtBaseDevice*>(device->UnderlyingDevice());
  if (pjrt_device == nullptr) {
    return errors::Internal(
        "Cannot get device metadata from non-PJRT device \"", device->name(),
        "\". GetMetadata must only be called on a device derived from "
        "PjRtBaseDevice. Either an internal bug has been triggered, or an "
        "XLA-specific op has been placed on the wrong device.");
  }
  return &pjrt_device->metadata_;
}

}  // namespace tensorflow
