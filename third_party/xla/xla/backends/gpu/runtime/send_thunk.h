/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk that performs a send operation.
class SendThunk : public CollectiveThunk {
 public:
  SendThunk(ThunkInfo thunk_info, const HloSendInstruction* instr,
            int64_t replica_count, int64_t partition_count,
            const Buffer& buffer);
  SendThunk(ThunkInfo thunk_info, const P2PConfig& config,
            std::shared_ptr<AsyncEvents> async_events, const Buffer& buffer,
            absl::string_view instr_name);

  absl::Status Initialize(const InitializeParams& params) override;

  static absl::StatusOr<std::unique_ptr<SendThunk>> FromProto(
      ThunkInfo thunk_info, const SendThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CollectiveThunk::AsyncEventsMap& async_events_map);

  absl::StatusOr<ThunkProto> ToProto() const override;

  const CollectiveConfig& config() const override { return config_.config; }

  const Buffer& buffer() const { return buffer_; }

  const P2PConfig& p2p_config() const { return config_; }

  BufferUses buffer_uses() const override {
    BufferUses uses{
        BufferUse::Read(buffer_.source_buffer.slice,
                        buffer_.source_buffer.shape),
        BufferUse::Write(buffer_.destination_buffer.slice,
                         buffer_.destination_buffer.shape),
    };
    return uses;
  }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;

 private:
  const P2PConfig config_;
  const Buffer buffer_;
  std::string hlo_name_;
};

absl::Status RunSend(DeviceBufferPair& buffer, se::Stream& stream,
                     Communicator& comm, int64_t current_id, int64_t target_id,
                     absl::string_view device_string);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_
