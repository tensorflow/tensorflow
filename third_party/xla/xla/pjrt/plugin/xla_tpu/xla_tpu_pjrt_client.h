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

#ifndef XLA_PJRT_PLUGIN_XLA_TPU_XLA_TPU_PJRT_CLIENT_H_
#define XLA_PJRT_PLUGIN_XLA_TPU_XLA_TPU_PJRT_CLIENT_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"

namespace xla {

// Public entry point to get an XLA:TPU PjRtClient with default options
absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtTpuClient(
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_TPU_XLA_TPU_PJRT_CLIENT_H_
