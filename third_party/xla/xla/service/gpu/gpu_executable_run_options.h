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

#include <functional>
#include <optional>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"

namespace xla::gpu {

// A callback to get a unique clique ids.
using CliqueIdCallback =  // NOLINT
    std::function<absl::StatusOr<CliqueIds>(const CliqueKey&)>;

// A user-defined timeout handler attached to XLA:GPU execution.
struct ExecutionTimeoutHandler {
  enum class Scope {
    // Monitors host-side execution: thunk execution and work submission to GPU
    // streams.
    kHost,
    // Monitors device work enqueued by the execution, measured from the end of
    // host dispatch until the device completes all enqueued work.
    kDevice,
  };

  // Called when XLA:GPU execution exceeds the handler timeout. Runs on the
  // HangWatchdog thread and must not block for long.
  using Callback = absl::AnyInvocable<void(absl::string_view action,
                                           absl::Duration timeout) &&>;

  static bool IsHost(const ExecutionTimeoutHandler& handler) {
    return handler.scope == Scope::kHost;
  }
  static bool IsDevice(const ExecutionTimeoutHandler& handler) {
    return handler.scope == Scope::kDevice;
  }

  Scope scope;
  absl::Duration timeout;
  Callback callback;
};

// GPU-specific executable options.
// We keep these separate from ExecutableRunOptions to avoid adding
// dependencies to ExecutableRunOptions.
class GpuExecutableRunOptions {
 public:
  // A mapping from local device ordinal to global device ID.
  using DeviceIdMap = absl::btree_map<LocalDeviceId, GlobalDeviceId>;

  // Sets a mapping from local device ordinals to global device IDs.
  // Used only on NVidia GPUs for cross-host NCCL collectives. If set, the
  // elements of `device_assignment` are interpreted as global device IDs, not
  // local device ordinals.
  GpuExecutableRunOptions& set_gpu_global_device_ids(
      std::optional<DeviceIdMap> device_ids);
  const std::optional<DeviceIdMap>& gpu_global_device_ids() const;

  // Callback that returns a unique clique id for a given clique key.
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

  // Sets a function that builds timeout handlers for each XLA:GPU execution.
  GpuExecutableRunOptions& set_execution_timeout_handlers(
      std::function<std::vector<ExecutionTimeoutHandler>()> handlers);

  // Returns timeout handlers for a new XLA:GPU execution, or an empty vector if
  // timeout handlers are not set.
  std::vector<ExecutionTimeoutHandler> execution_timeout_handlers() const;

 private:
  bool requires_exclusive_lock_on_gpu_ = false;
  bool enable_mock_collectives_ = false;
  std::optional<DeviceIdMap> gpu_global_device_ids_;
  CliqueIdCallback clique_id_callback_;
  GpuCollectives* collectives_ = nullptr;
  std::optional<absl::flat_hash_map<GlobalDeviceId, IncarnationId>>
      incarnations_;
  std::function<std::vector<ExecutionTimeoutHandler>()>
      execution_timeout_handlers_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
