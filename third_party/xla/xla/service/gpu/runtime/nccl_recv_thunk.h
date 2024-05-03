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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_RECV_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_RECV_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/nccl_p2p_thunk_common.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-recv.
class NcclRecvThunk : public NcclCollectiveThunk {
 public:
  NcclRecvThunk(ThunkInfo thunk_info, NcclApi* nccl_api,
                const HloRecvInstruction* instr, int64_t replica_count,
                int64_t partition_count, const Buffer& buffer);
  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclCommHandleWrapper comm_wrapper) override;
  AsyncStreamKind GetAsyncStreamKind() const override { return stream_kind_; }
  bool NeedFirstCallRendzevous() const override { return false; }

 private:
  const NcclP2PConfig config_;
  const Buffer buffer_;
  const AsyncStreamKind stream_kind_;
  std::shared_ptr<ExecutionCounters> execution_counters_;
};

absl::Status RunRecv(NcclApi* nccl_api,
                     NcclP2PConfig::SourceTargetMapEntry source_target,
                     DeviceBufferPair& buffer, se::Stream& stream,
                     NcclApi::NcclCommHandle comm,
                     absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_RECV_THUNK_H_
