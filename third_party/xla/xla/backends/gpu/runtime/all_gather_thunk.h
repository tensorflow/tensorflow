/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct AllGatherConfig {
  CollectiveConfig config;
};

// Thunk that performs an All-Gather among CUDA GPU-based replicas.
class AllGatherStartThunk : public CollectiveThunk {
 public:
  AllGatherStartThunk(ThunkInfo thunk_info, const HloAllGatherInstruction* inst,
                      std::vector<Buffer> buffers,
                      bool p2p_memcpy_enabled = false);

  static const char* GetHloOpName() { return "all-gather-start"; }

  static absl::Status CheckImplementable(const HloAllGatherInstruction* inst,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllGatherInstruction* inst);

  const CollectiveConfig& config() const override { return config_.config; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm) override;

 private:
  const AllGatherConfig config_;
  const std::vector<Buffer> buffers_;
};

absl::Status RunAllGather(std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_THUNK_H_
