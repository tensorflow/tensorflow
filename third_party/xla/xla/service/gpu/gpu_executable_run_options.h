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
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"

namespace xla::gpu {

// A callback to get a unique clique id.
//
// TODO(b/380457503): Delete this alias and switch to
// GpuCollectives::CliqueIdCallback.
using CliqueIdCallback =  // NOLINT
    std::function<absl::StatusOr<CliqueId>(const CliqueKey&)>;

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

  // Callback that returns a unique clieque id for a given clique key.
  GpuExecutableRunOptions& set_clique_id_callback(
      CliqueIdCallback clique_id_callback);
  const CliqueIdCallback& clique_id_callback() const;

  // Collectives API for running collective operations on the GPU devices.
  GpuExecutableRunOptions& set_collectives(GpuCollectives* collectives);
  GpuCollectives* collectives() const;

  // The incarnation of every device.
  GpuExecutableRunOptions& set_incarnations(
      absl::flat_hash_map<GlobalDeviceId, IncarnationId> incarnations);
  const std::optional<absl::flat_hash_map<GlobalDeviceId, IncarnationId>>&
  incarnations() const;

  // Whether the run requires an exclusive lock on the GPU.
  bool requires_exclusive_lock_on_gpu() const {
    return requires_exclusive_lock_on_gpu_;
  }

  // Require writers lock on the GPU.
  GpuExecutableRunOptions& set_requires_exclusive_lock_on_gpu() {
    requires_exclusive_lock_on_gpu_ = true;
    return *this;
  }

  bool enable_mock_collectives() const { return enable_mock_collectives_; }

  // Enables mocking nccl collective operations on the GPU.
  GpuExecutableRunOptions& set_enable_mock_collectives() {
    enable_mock_collectives_ = true;
    return *this;
  }

 private:
  bool requires_exclusive_lock_on_gpu_ = false;
  bool enable_mock_collectives_ = false;
  std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids_;
  CliqueIdCallback clique_id_callback_;
  GpuCollectives* collectives_;
  std::optional<absl::flat_hash_map<GlobalDeviceId, IncarnationId>>
      incarnations_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
