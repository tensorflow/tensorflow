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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/framework/device_factory.h"

namespace tensorflow {

class NextPluggableDeviceFactory : public DeviceFactory {
 public:
  explicit NextPluggableDeviceFactory(
      const std::string& device_type,
      const std::string& compilation_device_name)
      : api_(TfnpdApi()),
        device_type_(device_type),
        compilation_device_name_(compilation_device_name) {}

  Status ListPhysicalDevices(std::vector<string>* devices) override;

  Status CreateDevices(const SessionOptions& session_options,
                       const std::string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;

 private:
  const TFNPD_Api* api_;
  const std::string device_type_;
  const std::string compilation_device_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_FACTORY_H_
