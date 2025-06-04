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

#include "xla/backends/gpu/runtime/replica_id_thunk.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status ReplicaOrPartitionIdThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.collective_params->device_assn->LogicalIdForDevice(
                          global_device_id));
  int id = kind() == Kind::kReplicaId ? logical_id.replica_id
                                      : logical_id.computation_id;
  return params.stream->Memset32(&dest_addr, id, /*size=*/4);
}

absl::StatusOr<ThunkProto> ReplicaIdThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  auto* replica_id_thunk_proto = proto.mutable_replica_id_thunk();
  TF_ASSIGN_OR_RETURN(*replica_id_thunk_proto->mutable_dest_buffer(),
                      dest().ToProto());
  return proto;
}

/*static*/ absl::StatusOr<std::unique_ptr<ReplicaIdThunk>>
ReplicaIdThunk::FromProto(ThunkInfo thunk_info,
                          const ReplicaIdThunkProto& proto,
                          absl::Span<const BufferAllocation> allocations) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice dest,
      BufferAllocation::Slice::FromProto(proto.dest_buffer(), allocations));
  return std::make_unique<ReplicaIdThunk>(std::move(thunk_info), dest);
}

}  // namespace gpu
}  // namespace xla
