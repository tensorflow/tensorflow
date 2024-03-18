/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SHARDING_SERDES_H_
#define XLA_PYTHON_IFRT_SHARDING_SERDES_H_

#include <memory>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/statusor.h"

namespace xla {
namespace ifrt {

class Client;

// Options for deserializing shardings. Function referenced by `lookup_device`
// must remain valid during deserialization.
struct DeserializeShardingOptions
    : llvm::RTTIExtends<DeserializeShardingOptions, DeserializeOptions> {
  explicit DeserializeShardingOptions(
      DeviceList::LookupDeviceFunc lookup_device)
      : lookup_device(lookup_device) {}

  static char ID;  // NOLINT

  // Function that converts device ids to devices.
  DeviceList::LookupDeviceFunc lookup_device;
};

// Casts `DeserializeOptions` into `DeserializeShardingOptions`.
absl::StatusOr<std::unique_ptr<DeserializeShardingOptions>>
GetDeserializeShardingOptions(std::unique_ptr<DeserializeOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SHARDING_SERDES_H_
