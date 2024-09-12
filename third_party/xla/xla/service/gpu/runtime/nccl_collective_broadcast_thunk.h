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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_BROADCAST_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_BROADCAST_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
// Thunk that performs a NCCL-based collective broadcast.
class NcclCollectiveBroadcastStartThunk : public NcclCollectiveThunk {
 public:
  static absl::Status CheckImplementable(const HloInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectiveBroadcastInstruction* inst);

  const NcclCollectiveConfig& config() const override { return config_; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

  static const char* GetHloOpName() { return "collective-broadcast-start"; }

  NcclCollectiveBroadcastStartThunk(
      ThunkInfo thunk_info, NcclApi* nccl_api,
      const HloCollectiveBroadcastInstruction* instr,
      std::vector<Buffer> buffers);

 protected:
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclCommHandleWrapper comm_wrapper) override;

 private:
  const NcclCollectiveConfig config_;
  const std::vector<Buffer> buffers_;
};

absl::Status RunCollectiveBroadcast(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream,
                                    NcclApi::NcclCommHandle comm,
                                    NcclApi* nccl_api);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_COLLECTIVE_BROADCAST_THUNK_H_
