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

#include <map>
#include <optional>

#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique_key.h"

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

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
