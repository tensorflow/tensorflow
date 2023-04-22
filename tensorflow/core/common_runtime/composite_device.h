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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COMPOSITE_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COMPOSITE_DEVICE_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

extern const char* const kCompositeDeviceType;

// A virtual device which represents a set of devices. We don't execute any
// op on this virtial device.
class CompositeDevice : public Device {
 public:
  Status Sync() override {
    return errors::Internal(
        "Sync() should never been invoked on CompositeDevice.");
  }

  Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }

  const std::vector<string>* underlying_devices() const {
    return &underlying_devices_;
  }

  // Helper for creating a CompositeDevice on the same task as the given host
  // CPU.
  static std::unique_ptr<CompositeDevice> MakeDevice(
      const std::vector<string>& underlying_devices, const int unique_device_id,
      const DeviceNameUtils::ParsedName& host_name, Status* status);

  // Helper for creating a CompositeDevice with the given device name.
  static std::unique_ptr<CompositeDevice> MakeDevice(
      const std::vector<string>& underlying_devices, const string& device_name,
      Status* status);

  bool IsRemoteCallAllowed() const override { return false; }

 private:
  CompositeDevice(const DeviceAttributes& device_attributes,
                  const std::vector<string>& underlying_devices)
      : Device(/*env=*/nullptr, device_attributes),
        underlying_devices_(underlying_devices) {}

  const std::vector<string> underlying_devices_;

  TF_DISALLOW_COPY_AND_ASSIGN(CompositeDevice);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COMPOSITE_DEVICE_H_
