/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/replica_id_thunk.h"

namespace xla {
namespace gpu {

ReplicaIdThunk::ReplicaIdThunk(const BufferAllocation::Slice& dest,
                               const HloInstruction* instr)
    : Thunk(Kind::kReplicaId, instr), dest_(dest) {}

Status ReplicaIdThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  TF_ASSIGN_OR_RETURN(int replica_id,
                      params.device_assn->ReplicaIdForDeviceOrdinal(
                          params.stream->parent()->device_ordinal()));
  params.stream->ThenMemset32(&dest_addr, replica_id, /*size=*/4);
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
