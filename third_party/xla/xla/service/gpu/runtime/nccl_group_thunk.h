/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_GROUP_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_GROUP_THUNK_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

// NCCL group thunk fuses together a set of arbitrary operations into a single
// NCCL group call in order for them to be dispatched to NCCL as a NCCL group.
// NCCL may or may not execute them in parallel.

class NcclGroupThunk : public NcclCollectiveThunk {
 public:
  NcclGroupThunk(Thunk::Kind kind, ThunkInfo thunk_info, NcclApi* nccl_api,
                 const HloInstruction* instruction, int64_t replica_count,
                 int64_t partition_count);

 protected:
  const NcclCollectiveConfig& config() const override { return config_; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclCommHandleWrapper comm_wrapper) override;
  AsyncStreamKind GetAsyncStreamKind() const override;

 private:
  const NcclCollectiveConfig config_;
  NcclApi* nccl_api_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_GROUP_THUNK_H_
