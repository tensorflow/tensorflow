/* Copyright 2023 The OpenXLA Authors.

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
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/lockable.h"
#include "xla/stream_executor/stream.h"
#include "tsl/lib/gtl/int_type.h"

namespace xla {
namespace gpu {

TSL_LIB_GTL_DEFINE_INT_TYPE(OpId, int64_t);

struct NcclCommName {
  static std::string ToString(NcclApi::NcclCommHandle comm) {
    return absl::StrFormat("lockable comm %p", comm);
  }
};

struct NcclComm : public Lockable<NcclApi::NcclCommHandle, NcclCommName> {
  explicit NcclComm(NcclApi::NcclCommHandle comm) : Lockable(comm) {}
};

// Create the mock nccl communicator assuming all hosts have the same hardwares.
absl::StatusOr<NcclComm::Lock> LockMockNcclComm(
    const Thunk::CollectiveExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model);

absl::StatusOr<NcclComm::Lock> AcquireMockNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    std::vector<GlobalDeviceId> local_devices, size_t num_local_participants,
    const NcclCliqueIdCallback& clique_id_callback, int rank, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model);

// Mock a Nccl collective op including all-reduce, all-gather, and
// reduce-scatter.
absl::Status RunMockNcclCollectives(NcclApi* nccl_api,
                                    std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream,
                                    NcclApi::NcclCommHandle comm,
                                    Thunk::Kind reduce_op);

// Mock a NCCL-based All-To-All op.
absl::Status RunMockNcclAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream,
                                 NcclApi::NcclCommHandle comm);

// Mock a collective permute op.
absl::Status RunMockCollectivePermute(
    NcclApi* nccl_api, NcclP2PConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, NcclApi::NcclCommHandle comm,
    absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MOCK_NCCL_UTILS_H_
