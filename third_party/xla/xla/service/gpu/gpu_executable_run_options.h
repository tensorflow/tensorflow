/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_

#include <cstdint>
#include <map>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// GPU-specific executable options.
// We keep these separate from ExecutableRunOptions to avoid adding
// dependencies to ExecutableRunOptions.
class GpuExecutableRunOptions {
 public:
  // Sets a mapping from local device ordinals to global device IDs.
  // Used only on NVidia GPUs for cross-host NCCL collectives. If set, the
  // elements of `device_assignment` are interpreted as global device IDs, not
  // local device ordinals.
  GpuExecutableRunOptions& set_gpu_global_device_ids(
      std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids);
  const std::optional<std::map<int, GlobalDeviceId>>& gpu_global_device_ids()
      const;

  // Callback that returns a ncclUniqueId encoded as a string for a group of
  // communicating GPU devices. Used only on NVidia GPUs.
  GpuExecutableRunOptions& set_nccl_clique_id_callback(
      NcclCliqueIdCallback nccl_clique_id_callback);
  const NcclCliqueIdCallback& nccl_clique_id_callback() const;

  // Whether the run requires an exclusive lock on the GPU.
  bool requires_exclusive_lock_on_gpu() const {
    return requires_exclusive_lock_on_gpu_;
  }

  // Require writers lock on the GPU.
  GpuExecutableRunOptions& set_requires_exclusive_lock_on_gpu() {
    requires_exclusive_lock_on_gpu_ = true;
    return *this;
  }

  bool enable_mock_nccl_collectives() const {
    return enable_mock_nccl_collectives_;
  }

  // Enables mocking nccl collective operations on the GPU.
  GpuExecutableRunOptions& set_enable_mock_nccl_collectives() {
    enable_mock_nccl_collectives_ = true;
    return *this;
  }

  enum class MockNcclTopoModel { kGCPA3, kNvidia };
  // Gets the nccl network topology used in mocking calls.
  MockNcclTopoModel mock_nccl_topo_model() const {
    return mock_nccl_topo_model_;
  }
  GpuExecutableRunOptions& set_mock_nccl_topo_model(
      MockNcclTopoModel mock_nccl_topo_model) {
    mock_nccl_topo_model_ = mock_nccl_topo_model;
    return *this;
  }

 private:
  bool requires_exclusive_lock_on_gpu_ = false;
  bool enable_mock_nccl_collectives_ = false;
  MockNcclTopoModel mock_nccl_topo_model_ = MockNcclTopoModel::kGCPA3;
  std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids_;
  NcclCliqueIdCallback nccl_clique_id_callback_;
};

// NCCL-related execution parameters.
class NcclExecuteParams {
 public:
  // A mapping from local device ordinals to global device IDs.
  using GlobalDeviceIdMap = std::map<int32_t, GlobalDeviceId>;

  // Creates NCCL execution parameters from the run options for the given local
  // device. Returns an error if run options are misconfigured (i.e. missing a
  // global device mapping for a local device ordinal).
  static absl::StatusOr<NcclExecuteParams> Create(
      const ServiceExecutableRunOptions& run_options,
      int64_t local_device_ordinal);

  RunId run_id() const;

  int64_t local_device_ordinal() const;

  GlobalDeviceId global_device_id() const;

  const GlobalDeviceIdMap* global_device_id_map() const;

  const DeviceAssignment* device_assn() const;

  const NcclCliqueIdCallback* nccl_clique_id_callback() const;

 private:
  NcclExecuteParams(RunId run_id, int64_t local_device_ordinal,
                    GlobalDeviceId global_device_id,
                    const DeviceAssignment* device_assn,
                    const GlobalDeviceIdMap* global_device_id_map,
                    const NcclCliqueIdCallback* nccl_clique_id_callback);

  RunId run_id_;
  int64_t local_device_ordinal_;
  GlobalDeviceId global_device_id_;
  const DeviceAssignment* device_assn_;
  const GlobalDeviceIdMap* global_device_id_map_;
  const NcclCliqueIdCallback* nccl_clique_id_callback_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
