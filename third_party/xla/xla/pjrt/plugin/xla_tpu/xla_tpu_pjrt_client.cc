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

#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"

const char kTpuPjrtName[] = "tpu";

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtTpuClient(
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store) {
  return GetCApiClient(kTpuPjrtName, create_options, kv_store);
}

}  // namespace xla
