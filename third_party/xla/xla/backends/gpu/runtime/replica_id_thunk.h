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

#ifndef XLA_BACKENDS_GPU_RUNTIME_REPLICA_ID_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_REPLICA_ID_THUNK_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace gpu {

// Thunk that implements the ReplicaId(Idx == 0) or PartitionId(Idx == 1).
class ReplicaOrPartitionIdThunk : public Thunk {
 public:
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferAllocation::Slice dest() const { return dest_; }

 protected:
  ReplicaOrPartitionIdThunk(Kind kind, ThunkInfo thunk_info,
                            const BufferAllocation::Slice& dest)
      : Thunk(kind, thunk_info), dest_(dest) {}

 private:
  const BufferAllocation::Slice dest_;
};

class ReplicaIdThunk : public ReplicaOrPartitionIdThunk {
 public:
  static absl::StatusOr<std::unique_ptr<ReplicaIdThunk>> FromProto(
      ThunkInfo thunk_info, const ReplicaIdThunkProto& proto,
      absl::Span<const BufferAllocation> allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

  ReplicaIdThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : ReplicaOrPartitionIdThunk(Kind::kReplicaId, thunk_info, dest) {}
};

class PartitionIdThunk : public ReplicaOrPartitionIdThunk {
 public:
  static absl::StatusOr<std::unique_ptr<PartitionIdThunk>> FromProto(
      ThunkInfo thunk_info, const PartitionIdThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

  PartitionIdThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : ReplicaOrPartitionIdThunk(Kind::kPartitionId, thunk_info, dest) {}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_REPLICA_ID_THUNK_H_
