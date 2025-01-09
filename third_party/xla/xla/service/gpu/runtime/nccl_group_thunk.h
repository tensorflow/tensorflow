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

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

// NCCL group thunk fuses together a set of arbitrary operations into a single
// NCCL group call in order for them to be dispatched to NCCL as a NCCL group.
// NCCL may or may not execute them in parallel.

class NcclGroupThunk : public Thunk {
 public:
  NcclGroupThunk(const HloInstruction* instruction, Thunk::Kind kind,
                 std::vector<std::unique_ptr<Thunk>> thunks,
                 AsyncStreamKind stream_kind);
  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status ExecuteOnStream(const Thunk::ExecuteParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }

 private:
  ThunkSequence thunks_;
  AsyncStreamKind stream_kind_;
  std::shared_ptr<NcclCollectiveThunk::AsyncEvents> async_events_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_GROUP_THUNK_H_
