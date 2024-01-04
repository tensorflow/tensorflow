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

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/mock_nccl_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

struct MockNcclComm {};
void DestroyMockNcclComm::operator()(MockNcclComm_t mock_comm) {
  delete mock_comm;
}

StatusOr<MockNcclCommReference> InitializeMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization) {
  return Unimplemented("Initialize mock nccl comm is not implemented.");
}

Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, MockNcclComm_t mock_comm,
                              Thunk::Kind reduce_op) {
  return Unimplemented("Mock nccl collectives is not implemented.");
}

Status RunMockNcclAllToAll(bool has_split_dimension,
                           std::vector<DeviceBufferPair>& buffers,
                           se::Stream& stream, MockNcclComm_t mock_comm) {
  return Unimplemented("Mock nccl AllToAll is not implemented.");
}

Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, MockNcclComm_t mock_comm,
    absl::string_view device_string, int64_t current_id) {
  return Unimplemented("Mock collective permute is not implemented.");
}

}  // namespace gpu
}  // namespace xla
