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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
// Caches some miscellaneous information about TF devices.  Thread compatible.
class DeviceInfoCache {
 public:
  xla::StatusOr<const XlaOpRegistry::DeviceRegistration*> GetCompilationDevice(
      absl::string_view device_name);
  xla::StatusOr<std::reference_wrapper<const DeviceType>> GetDeviceTypeFor(
      absl::string_view device_name);

 private:
  absl::flat_hash_map<string, const XlaOpRegistry::DeviceRegistration*>
      device_to_device_registration_;
  absl::flat_hash_map<string, std::unique_ptr<DeviceType>>
      device_to_device_type_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
