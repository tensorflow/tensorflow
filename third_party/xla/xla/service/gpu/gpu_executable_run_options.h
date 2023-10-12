/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/service/global_device_id.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Key for naming up a particular NCCL clique.  This is just a set of unique
// device IDs (i.e. GPU IDs) and a stream_id. The device IDs must be global
// within a cluster. The stream_id is used to create different NCCL clique and
// communicators for collectives executed on different streams within an
// executable.
class NcclCliqueKey {
 public:
  explicit NcclCliqueKey(std::vector<GlobalDeviceId> devices,
                         int64_t stream_id = 0)
      : devices_(std::move(devices)), stream_id_(stream_id) {}

  template <typename H>
  friend H AbslHashValue(H h, const NcclCliqueKey& k) {
    return H::combine(std::move(h), k.devices_, k.stream_id_);
  }
  friend bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b) {
    return a.devices_ == b.devices_ && a.stream_id_ == b.stream_id_;
  }

  const std::vector<GlobalDeviceId>& devices() const { return devices_; }

  std::string ToString() const {
    return absl::StrCat("stream[", stream_id_, "]",
                        GlobalDeviceIdsToString(devices_));
  }

 private:
  const std::vector<GlobalDeviceId> devices_;
  const int64_t stream_id_;
};

using NcclUniqueIdCallback =
    std::function<StatusOr<std::string>(const NcclCliqueKey&)>;

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
  GpuExecutableRunOptions& set_nccl_unique_id_callback(
      NcclUniqueIdCallback nccl_unique_id_callback);
  const NcclUniqueIdCallback& nccl_unique_id_callback() const;

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

  // Enable mocking nccl collective operations on the GPU
  GpuExecutableRunOptions& set_enable_mock_nccl_collectives() {
    enable_mock_nccl_collectives_ = true;
    return *this;
  }

 private:
  bool requires_exclusive_lock_on_gpu_ = false;
  bool enable_mock_nccl_collectives_ = false;
  std::optional<std::map<int, GlobalDeviceId>> gpu_global_device_ids_;
  NcclUniqueIdCallback nccl_unique_id_callback_;
};

// NCCL-related execution parameters.
struct NcclExecuteParams {
  NcclExecuteParams(const ServiceExecutableRunOptions& run_options,
                    se::StreamExecutor* stream_executor);

  se::StreamExecutor* stream_executor;
  RunId run_id;
  const DeviceAssignment* device_assn;                         // never null
  const std::map<int, GlobalDeviceId>* gpu_global_device_ids;  // may be null
  const NcclUniqueIdCallback* nccl_unique_id_callback;         // may be null

  StatusOr<GlobalDeviceId> GetGlobalDeviceId() const;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_RUN_OPTIONS_H_
