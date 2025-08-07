/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_SEND_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_SEND_THUNK_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// Thunk to perform NVSHMEM send operations
class NvshmemSendThunk : public NvshmemCollectiveThunk {
 public:
  NvshmemSendThunk(ThunkInfo thunk_info, const HloSendInstruction* inst,
                   int64_t replica_count, int64_t partition_count,
                   const CollectiveThunk::Buffer& buffer,
                   std::shared_ptr<NvshmemBufferAddresses> buffer_addresses);
  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                    se::Stream& stream) override;

 private:
  const P2PConfig config_;
  const CollectiveThunk::Buffer buffer_;
  std::unique_ptr<ExecutionCounters> execution_counters_;
  std::string hlo_name_;
  std::shared_ptr<NvshmemBufferAddresses> buffer_addresses_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_SEND_THUNK_H_
