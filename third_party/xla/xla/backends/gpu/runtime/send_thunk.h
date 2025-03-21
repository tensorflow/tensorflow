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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk that performs a send operation.
class SendThunk : public CollectiveThunk {
 public:
  SendThunk(ThunkInfo thunk_info, const HloSendInstruction* instr,
            int64_t replica_count, int64_t partition_count,
            const Buffer& buffer);
  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::Status RunCollective(const ExecuteParams& params, se::Stream& stream,
                             CommunicatorHandle comm_handle) override;
  bool NeedFirstCallRendzevous() const override { return false; }

 private:
  const P2PConfig config_;
  const Buffer buffer_;
  std::shared_ptr<ExecutionCounters> execution_counters_;
  std::string hlo_name_;
};

absl::Status RunSend(GpuCollectives* collectives,
                     P2PConfig::SourceTargetMapEntry source_target,
                     DeviceBufferPair& buffer, se::Stream& stream,
                     Communicator* comm, absl::string_view device_string,
                     int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_SEND_THUNK_H_
