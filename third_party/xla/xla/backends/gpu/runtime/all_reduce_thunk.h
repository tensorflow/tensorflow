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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct AllReduceConfig {
  CollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class AllReduceReduceScatterThunkBase : public CollectiveThunk {
 public:
  AllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                  AllReduceConfig config,
                                  std::vector<Buffer> buffers, bool is_sync);

  const CollectiveConfig& config() const override { return config_.config; }
  ReductionKind reduction_kind() const { return config_.reduction_kind; }

  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  const AllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

// -----------------------------------------------------------------------------
// AllReduce thunk.
// -----------------------------------------------------------------------------

class AllReduceStartThunk : public AllReduceReduceScatterThunkBase {
 public:
  AllReduceStartThunk(ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
                      std::vector<Buffer> buffers,
                      bool p2p_memcpy_enabled = false);

  static const char* GetHloOpName() { return "all-reduce-start"; }

  static absl::Status CheckImplementable(const HloAllReduceInstruction* inst,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllReduceInstruction* inst);

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm) override;

 private:
  CollectiveKernelThunk collective_kernel_thunk_;
};

// -----------------------------------------------------------------------------
// ReduceScatter thunk
// -----------------------------------------------------------------------------

class ReduceScatterStartThunk : public AllReduceReduceScatterThunkBase {
 public:
  ReduceScatterStartThunk(ThunkInfo thunk_info,
                          const HloReduceScatterInstruction* inst,
                          std::vector<Buffer> buffers,
                          bool p2p_memcpy_enabled = false);

  static const char* GetHloOpName() { return "reduce-scatter-start"; }

  static absl::Status CheckImplementable(
      const HloReduceScatterInstruction* inst, int64_t replica_count,
      int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloReduceScatterInstruction* inst);

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm) override;
};

// -----------------------------------------------------------------------------

absl::Status RunAllReduce(ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm);

absl::Status RunReduceScatter(ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_THUNK_H_
