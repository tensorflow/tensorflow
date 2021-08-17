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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REPLICA_ID_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REPLICA_ID_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

// Thunk that implements the ReplicaId(Idx == 0) or PartitionId(Idx == 1).
class ReplicaOrPartitionIdThunk : public Thunk {
  Status ExecuteOnStream(const ExecuteParams& params) override;

 protected:
  ReplicaOrPartitionIdThunk(Kind kind, ThunkInfo thunk_info,
                            const BufferAllocation::Slice& dest)
      : Thunk(kind, thunk_info), dest_(dest) {}

 private:
  const BufferAllocation::Slice dest_;
};

class ReplicaIdThunk : public ReplicaOrPartitionIdThunk {
 public:
  ReplicaIdThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : ReplicaOrPartitionIdThunk(Kind::kReplicaId, thunk_info, dest) {}
};

class PartitionIdThunk : public ReplicaOrPartitionIdThunk {
 public:
  PartitionIdThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : ReplicaOrPartitionIdThunk(Kind::kPartitionId, thunk_info, dest) {}
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REPLICA_ID_THUNK_H_
