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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct NcclAllToAllConfig {
  NcclCollectiveConfig config;
  bool has_split_dimension;
};

// Thunk that performs a NCCL-based All-to-All among CUDA GPU-based replicas.
class NcclAllToAllStartThunk : public NcclCollectiveThunk {
 public:
  NcclAllToAllStartThunk(ThunkInfo thunk_info, NcclApi* nccl_api,
                         const HloAllToAllInstruction* instr,
                         std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-to-all
  // call.
  static absl::Status CheckImplementable(const HloAllToAllInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static const char* GetHloOpName() { return "all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllToAllInstruction* instr);

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclApi::NcclCommHandle comm) override;

 private:
  const NcclAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
};

absl::Status RunAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, NcclApi::NcclCommHandle comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_
