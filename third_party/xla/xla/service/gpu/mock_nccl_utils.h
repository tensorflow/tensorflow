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

#ifndef XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_
#define XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Create the mock nccl communicator assuming all hosts have the same hardwares.
StatusOr<NcclComm::Lock> LockMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization);

StatusOr<NcclComm::Lock> AcquireMockNcclComm(
    ncclComm_t local_comm, RunId run_id, OpId op_id,
    std::vector<GlobalDeviceId> participants,
    std::vector<GlobalDeviceId> local_devices, size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank, int64_t stream_id,
    bool enable_clique_optimization);

// Mock a Nccl collective op including all-reduce, all-gather, and
// reduce-scatter.
Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, ncclComm_t comm,
                              Thunk::Kind reduce_op);

// Mock a NCCL-based All-To-All op.
Status RunMockNcclAllToAll(bool has_split_dimension,
                           std::vector<DeviceBufferPair>& buffers,
                           se::Stream& stream, ncclComm_t comm);

// Mock a collective permute op.
Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, ncclComm_t comm, absl::string_view device_string,
    int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_
