/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_BROADCAST_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_BROADCAST_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

struct CollectiveBroadcastMetadata {
  int64_t num_roots;
  std::unique_ptr<se::MemoryAllocation> bcast_roots = nullptr;
};
// Thunk that performs a collective broadcast.
class CollectiveBroadcastThunk : public CollectiveThunk {
 public:
  static absl::Status CheckImplementable(const HloInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectiveBroadcastInstruction* inst);

  const CollectiveConfig& config() const override { return config_; }

  static absl::string_view GetHloOpName() {
    return "collective-broadcast-start";
  }

  CollectiveBroadcastThunk(ThunkInfo thunk_info,
                           const HloCollectiveBroadcastInstruction* instr,
                           std::vector<Buffer> buffers, int devices_per_host,
                           bool p2p_memcpy_enabled = false,
                           bool has_dynamic_root = false);
  CollectiveBroadcastThunk(ThunkInfo thunk_info, CollectiveConfig config,
                           std::vector<Buffer> buffers, int devices_per_host,
                           bool has_dynamic_root = false);
  absl::Status Initialize(const InitializeParams& params) override;

  static absl::StatusOr<std::unique_ptr<CollectiveBroadcastThunk>> FromProto(
      ThunkInfo thunk_info, const CollectiveBroadcastThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      int devices_per_host);

  absl::StatusOr<ThunkProto> ToProto() const override;

 protected:
  bool RequiresRendezvous() const override { return true; }

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

 private:
  const CollectiveConfig config_;
  PerDeviceState<CollectiveBroadcastMetadata> per_device_cb_metadata_;
  bool has_dynamic_root_;
};

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator& comm,
                                    CollectiveBroadcastMetadata* cb_metadata,
                                    bool has_dynamic_root = false);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_BROADCAST_THUNK_H_
