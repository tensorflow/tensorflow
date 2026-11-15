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

#include "xla/backends/gpu/target_config/cudnn_device_props.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/backends/gpu/target_config/cudnn_device_props_json.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

absl::StatusOr<std::shared_ptr<cudnn_frontend::DeviceProperties>>
BuildDeviceProperties(const stream_executor::DeviceDescription& desc) {
  std::string json = FormatDevicePropertiesJson(desc);
  auto device_props = std::make_shared<cudnn_frontend::DeviceProperties>();
  RETURN_IF_CUDNN_FRONTEND_ERROR(device_props->deserialize(
      std::vector<uint8_t>(json.begin(), json.end())));
  return device_props;
}

}  // namespace xla::gpu
