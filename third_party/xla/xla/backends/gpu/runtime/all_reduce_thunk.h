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
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/event.h"
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

  absl::StatusOr<bool> ShouldUseOneShotAllReduceKernel(
      const GpuCliqueKey& clique_key,
      const CollectiveCliques* collective_cliques);

  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  absl::Status RunCollective(const ExecuteParams& params, se::Stream& stream,
                             CommunicatorHandle comm_handle) override;

 private:
  bool one_shot_kernel_enabled_ = false;

  absl::Mutex mutex_;

  // Local buffer allocations to copy input data for the one-shot kernel.
  absl::flat_hash_map<se::StreamExecutor*, se::DeviceMemoryHandle>
      local_buffer_allocs_ ABSL_GUARDED_BY(mutex_);

  // Events to synchronize steams on different devices at the start of the
  // one-shot kernel.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Event>>
      start_events_ ABSL_GUARDED_BY(mutex_);

  // Events to synchronize steams on different devices at the end of the
  // one-shot kernel.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Event>>
      end_events_ ABSL_GUARDED_BY(mutex_);
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
  absl::Status RunCollective(const ExecuteParams& params, se::Stream& stream,
                             CommunicatorHandle comm_handle) override;
};

// -----------------------------------------------------------------------------

absl::Status RunAllReduce(GpuCollectives* collectives,
                          ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm);

absl::Status RunReduceScatter(GpuCollectives* collectives,
                              ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator* comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_THUNK_H_
