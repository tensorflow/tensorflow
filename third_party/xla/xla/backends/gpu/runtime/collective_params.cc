/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_params.h"

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

using GlobalDeviceIdMap = CollectiveParams::GlobalDeviceIdMap;

// Returns global device id for a local device ordinal or an error if global
// device id map is misconfigured and missing an entry for a local device.
static absl::StatusOr<GlobalDeviceId> GetGlobalDeviceId(
    const GlobalDeviceIdMap* device_id_map, LocalDeviceId local_device_id) {
  // No local -> global mapping was provided; assume the identity mapping.
  if (!device_id_map) {
    return GlobalDeviceId(local_device_id.value());
  }

  // Find a global device id in a global device id map.
  auto it = device_id_map->find(local_device_id);
  if (it == device_id_map->end()) {
    return NotFound("No global device id found for local device ordinal: %d",
                    local_device_id.value());
  }

  return it->second;
}

absl::StatusOr<CollectiveParams> CollectiveParams::Create(
    const ServiceExecutableRunOptions& run_options,
    absl::Span<se::Stream* const> async_streams, LocalDeviceId local_device_id,
    int64_t collective_max_nchannels, int64_t p2p_max_nchannels) {
  const GpuExecutableRunOptions* gpu_options =
      run_options.run_options().gpu_executable_run_options();

  const std::string& platform_name =
      run_options.run_options().stream()->parent()->GetPlatform()->Name();
  auto* collectives = gpu_options && gpu_options->collectives()
                          ? gpu_options->collectives()
                          : GpuCollectives::Default(platform_name);

  auto* device_id_map = gpu_options && gpu_options->gpu_global_device_ids()
                            ? &*gpu_options->gpu_global_device_ids()
                            : nullptr;

  auto* clique_id_callback = gpu_options && gpu_options->clique_id_callback()
                                 ? &gpu_options->clique_id_callback()
                                 : nullptr;

  auto* incarnations = gpu_options && gpu_options->incarnations().has_value()
                           ? &*gpu_options->incarnations()
                           : nullptr;

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      GetGlobalDeviceId(device_id_map, local_device_id));

  return CollectiveParams(collectives, run_options.stream()->parent(),
                          run_options.run_options().run_id(), async_streams,
                          local_device_id, global_device_id,
                          run_options.run_options().device_assignment(),
                          device_id_map, clique_id_callback, incarnations,
                          collective_max_nchannels, p2p_max_nchannels);
}

CollectiveParams::CollectiveParams(
    GpuCollectives* collectives, se::StreamExecutor* executor, RunId run_id,
    absl::Span<se::Stream* const> async_streams, LocalDeviceId local_device_id,
    GlobalDeviceId global_device_id, const DeviceAssignment* device_assn,
    const GlobalDeviceIdMap* global_device_id_map,
    const CliqueIdCallback* nccl_clique_id_callback,
    const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations,
    int64_t collective_max_nchannels, int64_t p2p_max_nchannels)
    : collectives(collectives),
      executor(executor),
      run_id(run_id),
      async_streams(async_streams.begin(), async_streams.end()),
      local_device_id(local_device_id),
      global_device_id(global_device_id),
      device_assn(device_assn),
      global_device_id_map(global_device_id_map),
      nccl_clique_id_callback(nccl_clique_id_callback),
      incarnations(incarnations),
      collective_max_nchannels(collective_max_nchannels),
      p2p_max_nchannels(p2p_max_nchannels) {}

}  // namespace xla::gpu
