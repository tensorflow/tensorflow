/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/mock_nccl_utils.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

absl::StatusOr<NcclComm::Lock> AcquireMockNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    std::vector<GlobalDeviceId> local_devices, size_t num_local_participants,
    const NcclCliqueIdCallback& clique_id_callback, int rank, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model) {
  return Unimplemented("AcquireMockNcclComm is not implemented.");
}

absl::StatusOr<NcclComm::Lock> LockMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model) {
  return Unimplemented("LockMockNcclComm is not implemented.");
}

absl::Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream, ncclComm_t mock_comm,
                                    Thunk::Kind reduce_op) {
  return Unimplemented("Mock nccl collectives is not implemented.");
}

absl::Status RunMockNcclAllToAll(bool has_split_dimension,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream, ncclComm_t mock_comm) {
  return Unimplemented("Mock nccl AllToAll is not implemented.");
}

absl::Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, ncclComm_t mock_comm, absl::string_view device_string,
    int64_t current_id) {
  return Unimplemented("Mock collective permute is not implemented.");
}

}  // namespace gpu
}  // namespace xla
