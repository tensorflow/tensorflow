/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_INTERCONNECT_RESOURCE_H_
#define XLA_STREAM_EXECUTOR_DEVICE_INTERCONNECT_RESOURCE_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

class DeviceInterconnectResource : public StreamExecutor::Resource {
 public:
  using InfoMap = absl::flat_hash_map<int32_t, DeviceInterconnectInfo>;

  // Constructor that accepts the shared map
  explicit DeviceInterconnectResource(std::shared_ptr<const InfoMap> map)
      : device_interconnect_info_map_(std::move(map)) {}

  // Const getter for the map
  const InfoMap& interconnect_map() const {
    return *device_interconnect_info_map_;
  }

 private:
  std::shared_ptr<const InfoMap> device_interconnect_info_map_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_INTERCONNECT_RESOURCE_H_
