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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk that performs a NVSHMEM-based collective permute.
class NvshmemCollectivePermuteStartThunk : public NvshmemCollectiveThunk {
 public:
  NvshmemCollectivePermuteStartThunk(
      ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
      int64_t replica_count, int64_t partition_count,
      const std::vector<CollectiveThunk::Buffer>& buffers,
      bool p2p_memcpy_enabled = false,
      AsyncStreamKind stream_kind = AsyncStreamKind::kCollective);

  static const char* GetHloOpName() { return "collective-permute-start"; }

  static absl::Status CheckImplementable(
      const HloCollectivePermuteInstruction* inst, int64_t replica_count,
      int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  static P2PConfig GetNvshmemP2PConfig(
      const HloCollectivePermuteInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                    se::Stream& stream) override;

 private:
  const P2PConfig config_;
  const std::vector<CollectiveThunk::Buffer> buffers_;
  const bool p2p_memcpy_enabled_ = false;
};

// Thunk that performs a NVSHMEM-based collective permute done operation.
class NvshmemCollectivePermuteDoneThunk : public NvshmemCollectiveDoneThunk {
 public:
  NvshmemCollectivePermuteDoneThunk(
      ThunkInfo thunk_info,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events,
      AsyncStreamKind stream_kind);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
};

absl::Status RunCollectivePermute(P2PConfig::SourceTargetMapEntry source_target,
                                  std::vector<DeviceBufferPair>& buffers,
                                  se::Stream& stream,
                                  absl::string_view device_string,
                                  int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_
