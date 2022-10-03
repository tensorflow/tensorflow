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

Status ReplicaOrPartitionIdThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.nccl_params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      params.nccl_params.device_assn->LogicalIdForDevice(global_device_id));
  int id = kind() == Kind::kReplicaId ? logical_id.replica_id
                                      : logical_id.computation_id;
  params.stream->ThenMemset32(&dest_addr, id, /*size=*/4);
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
