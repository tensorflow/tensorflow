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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_BACKEND_ASSIGNER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_BACKEND_ASSIGNER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// CollectiveBackendAssigner is a pass that assigns the appropriate backend
// (NVSHMEM or DEFAULT) to collective operations based on:
// 1. Communication pattern: Uses NVSHMEM for intranode communication
// 2. Message size: Uses NVSHMEM for messages smaller than threshold_in_bytes
// This pass helps optimize collective operations by choosing the most efficient
// backend based on the operation's characteristics.

constexpr int64_t kDefaultThresholdInBytes = 16 * 1024 * 1024;  // 16MB

class CollectiveBackendAssigner : public HloModulePass {
 public:
  explicit CollectiveBackendAssigner(
      const se::GpuComputeCapability& gpu_version,
      int num_visible_devices_per_process,
      int64_t threshold_in_bytes = kDefaultThresholdInBytes)
      : gpu_version_(gpu_version),
        num_visible_devices_per_process_(num_visible_devices_per_process),
        threshold_in_bytes_(threshold_in_bytes) {}

  absl::string_view name() const override {
    return "collective-backend-assigner";
  }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
  int num_visible_devices_per_process_;
  int64_t threshold_in_bytes_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_BACKEND_ASSIGNER_H_
