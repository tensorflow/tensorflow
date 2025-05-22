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

#ifndef XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_
#define XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"

namespace xla {

// Options for creating a XLA:GPU PjRtClient.
struct GpuClientOptions {
  GpuAllocatorConfig allocator_config;

  int node_id = 0;

  int num_nodes = 1;

  std::optional<std::set<int>> allowed_devices = std::nullopt;

  std::optional<std::string> platform_name = std::nullopt;

  bool should_stage_host_to_device_transfers = true;

  // kv_store must be non-null if num_nodes > 1.
  std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;

  bool enable_mock_nccl = false;

  std::optional<std::string> mock_gpu_topology;

  std::optional<int> slice_index;

  bool use_tfrt_gpu_client = false;
};

}  //  namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_
