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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

namespace tensorflow {
namespace tpu {
namespace {

class TpuSystemDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status TpuSystemDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  int device_count = 0;
  TF_RETURN_IF_ERROR(TpuPlatform::TpusPerHost(&device_count));
  if (device_count == 0) {
    VLOG(1) << "Host has no TPUs, not creating a TPU_SYSTEM device";
    return Status::OK();
  }

  devices->push_back("/physical_device:TPU_SYSTEM:0");

  return Status::OK();
}

Status TpuSystemDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  int device_count = 0;
  TF_RETURN_IF_ERROR(TpuPlatform::TpusPerHost(&device_count));
  if (device_count == 0) {
    VLOG(1) << "Host has no TPUs, not creating a TPU_SYSTEM device";
    return Status::OK();
  }

  int64 memory_limit;
  TF_RETURN_IF_ERROR(TpuPlatform::TpuMemoryLimit(&memory_limit));

  // Creates a device that represents a Jellyfish distributed system.
  const DeviceAttributes attrs = Device::BuildDeviceAttributes(
      strings::StrCat(name_prefix, "/device:", DEVICE_TPU_SYSTEM, ":", 0),
      DeviceType(DEVICE_TPU_SYSTEM), Bytes(memory_limit), DeviceLocality(),
      strings::StrCat("device: ", DEVICE_TPU_SYSTEM, " device"));
  devices->push_back(absl::make_unique<VirtualDevice>(options.env, attrs));
  VLOG(1) << "Created TPU_SYSTEM device. This host has " << device_count
          << " TPUs";

  return Status::OK();
}

}  // namespace

void RegisterTpuSystemDevice() {
  REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_TPU_SYSTEM, TpuSystemDeviceFactory);
}

}  // namespace tpu
}  // namespace tensorflow
