#include <memory>

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_data.pb.h"
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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Thunk that performs a collective broadcast.
class CollectiveBroadcastStartThunk : public CollectiveThunk {
 public:
  static absl::Status CheckImplementable(const HloInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectiveBroadcastInstruction* inst);

  const CollectiveConfig& config() const override { return config_; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

  static absl::string_view GetHloOpName() {
    return "collective-broadcast-start";
  }

  CollectiveBroadcastStartThunk(ThunkInfo thunk_info,
                                const HloCollectiveBroadcastInstruction* instr,
                                std::vector<Buffer> buffers,
                                bool p2p_memcpy_enabled = false);
  CollectiveBroadcastStartThunk(ThunkInfo thunk_info, CollectiveConfig config,
                                std::shared_ptr<AsyncEvents> async_events,
                                std::vector<Buffer> buffers);

  static absl::StatusOr<std::unique_ptr<CollectiveBroadcastStartThunk>>
  FromProto(ThunkInfo thunk_info,
            const CollectiveBroadcastStartThunkProto& thunk_proto,
            absl::Span<const BufferAllocation> buffer_allocations,
            CollectiveThunk::AsyncEventsMap& async_events_map);

  absl::StatusOr<ThunkProto> ToProto() const override;

  BufferUses buffer_uses() const override {
    BufferUses uses;
    uses.reserve(buffers_.size() * 2);
    for (const Buffer& buffer : buffers_) {
      uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                     buffer.source_buffer.shape));
      uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                      buffer.destination_buffer.shape));
    }
    return uses;
  }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;

 private:
  const CollectiveConfig config_;
  const std::vector<Buffer> buffers_;
};

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, Communicator& comm);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_BROADCAST_THUNK_H_
