/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtCpuClient(
    CpuClientOptions options) {
  absl::flat_hash_map<std::string, PjRtValueType> create_options;
  create_options["asynchronous"] = options.asynchronous;
  if (options.cpu_device_count.has_value()) {
    create_options["cpu_device_count"] =
        static_cast<int64_t>(options.cpu_device_count.value());
  }
  create_options["max_inflight_computations_per_device"] =
      static_cast<int64_t>(options.max_inflight_computations_per_device);
  return GetCApiClient(kCpuPjrtName, create_options);
}

}  // namespace xla
