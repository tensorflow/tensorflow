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

#include "absl/time/time.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"

namespace xla {

class DistributedRuntimeClient;

// Options for creating a XLA:GPU PjRtClient.
struct GpuClientOptions {
  GpuAllocatorConfig allocator_config;

  int node_id = 0;

  int num_nodes = 1;

  std::optional<std::set<int>> allowed_devices = std::nullopt;

  std::optional<std::string> platform_name = std::nullopt;

  bool should_stage_host_to_device_transfers = true;

  // Optional factory for a host memory allocator to use for transfer. Used only
  // if `should_stage_host_to_device_transfers` is true.
  HostMemoryAllocator::Factory host_memory_allocator_factory;

  // kv_store must be non-null if num_nodes > 1.
  std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;

  // Optional distributed runtime client used to report execution timeouts to
  // the coordination service when abort_collectives_on_failure is enabled.
  std::shared_ptr<DistributedRuntimeClient> distributed_client = nullptr;

  // If true, aborts local collectives when the coordination service reports
  // that a task failed or restarted with a new incarnation.
  bool abort_collectives_on_failure = false;

  // If host execution or device work enqueued by an XLA:GPU execution does not
  // complete within this timeout, the client aborts all local collectives. Task
  // failure is detected and reported by the coordination service (e.g. missed
  // heartbeats).
  //
  // Can be used together with `xla_gpu_execution_terminate_timeout` and
  // `xla_gpu_device_execution_terminate_timeout`. Timeouts must be far apart
  // (e.g. abort at 5 minutes and terminate at 10 minutes) to give NCCL time to
  // abort collectives and the client to recover.
  absl::Duration abort_collectives_timeout = absl::InfiniteDuration();

  bool enable_mock_nccl = false;

  std::optional<std::string> mock_gpu_topology;

  std::optional<int> partition_index;

  bool use_tfrt_gpu_client = false;

  std::optional<bool> use_async_dispatch;

  std::optional<int> max_inflight_computations = 32;

  bool verify_topology_fingerprint = true;
};

}  //  namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_
