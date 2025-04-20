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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"

namespace xla::gpu {

// NVIDIA NVSHMEM library
class NvshmemCollectives : public GpuCollectives {
 public:
  ~NvshmemCollectives() override;

  static NvshmemCollectives* Default();

  void SetEnvInfo(int process_id, size_t num_processes,
                  size_t device_count_per_process,
                  std::weak_ptr<KeyValueStoreInterface> kv_store);

  absl::StatusOr<void*> Allocate(uint64_t bytes) final;

  absl::Status Deallocate(void* buffer) final;

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status GroupStart() final {
    return absl::UnimplementedError("Not implemented.");
  }
  absl::Status GroupEnd() final {
    return absl::UnimplementedError("Not implemented.");
  }

  bool IsImplemented() const final { return true; }

  bool IsGlobalConfig() const final { return false; }

  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback* clique_id_callback, bool is_local) final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Collectives::Config& config) {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, const Collectives::Config& config) final {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status InitializeTopology(Topology topology) final;

 private:
  absl::Status InitializeOnce();

  void Finalize();

  int process_id_ = -1;
  size_t num_processes_ = 0;
  size_t device_count_per_process_ = 0;
  std::weak_ptr<KeyValueStoreInterface> kv_store_;
  bool initialized_ = false;

  static constexpr char kKvStoreKey[] = "nvshmem_global_init";
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NVSHMEM_COLLECTIVES_H_
