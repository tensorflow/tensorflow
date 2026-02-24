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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// Parameters capturing all the details required for collective execution of
// XLA executables (multiple partitions and replicas).
struct CollectiveParams {
  // A mapping from local device ordinals to global device IDs.
  using GlobalDeviceIdMap = GpuExecutableRunOptions::DeviceIdMap;

  // Creates NCCL execution parameters from the run options for the given
  // local device. Returns an error if run options are misconfigured (i.e.
  // missing a global device mapping for a local device ordinal).
  static absl::StatusOr<CollectiveParams> Create(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<se::Stream* const> async_streams,
      LocalDeviceId local_device_id, int64_t collective_max_nchannels = 0,
      int64_t p2p_max_nchannels = 0);

  GpuCollectives* collectives;
  se::StreamExecutor* executor;

  // XLA execution run id allows us to distinguish collective operations
  // from different concurrent executions and avoid deadlocks.
  RunId run_id;

  // Streams for asynchronous collective communications.
  absl::InlinedVector<se::Stream*, 4> async_streams;

  LocalDeviceId local_device_id;
  GlobalDeviceId global_device_id;

  const DeviceAssignment* device_assn;
  const GlobalDeviceIdMap* global_device_id_map;
  const CliqueIdCallback* clique_id_callback;
  const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations;

  int64_t collective_max_nchannels;
  int64_t p2p_max_nchannels;

  int local_device_count = 0;

 private:
  CollectiveParams(
      GpuCollectives* collectives, se::StreamExecutor* executor, RunId run_id,
      absl::Span<se::Stream* const> async_streams,
      LocalDeviceId local_device_id, GlobalDeviceId global_device_id,
      const DeviceAssignment* device_assn,
      const GlobalDeviceIdMap* global_device_id_map,
      const CliqueIdCallback* clique_id_callback,
      const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations,
      int64_t collective_max_nchannels, int64_t p2p_max_nchannels,
      int local_device_count);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_
