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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
class PluggableDeviceFactory : public DeviceFactory {
 public:
  PluggableDeviceFactory(const string& device_type,
                         const string& platform_name);
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options,
                       const std::string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
  Status GetDeviceDetails(int device_index,
                          std::unordered_map<string, string>* details) override;

 private:
  // Populates *device_localities with the DeviceLocality descriptor for
  // every TfDeviceId.
  Status GetDeviceLocalities(int num_tf_devices,
                             std::vector<DeviceLocality>* device_localities);
  // Create a PluggableDevice associated with 'tf_device_id', allocates
  // (strictly) 'memory_limit' bytes of PluggableDevice memory to it, and adds
  // it to the 'devices' vector.
  Status CreatePluggableDevice(const SessionOptions& options,
                               const std::string& name_prefix,
                               TfDeviceId tf_device_id, int64 memory_limit,
                               const DeviceLocality& dev_locality,
                               std::vector<std::unique_ptr<Device>>* devices);

  const string device_type_;
  const string platform_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_FACTORY_H_
