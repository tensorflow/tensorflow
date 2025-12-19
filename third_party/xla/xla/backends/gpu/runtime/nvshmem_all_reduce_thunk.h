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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_ALL_REDUCE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_ALL_REDUCE_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Thunk that performs a nvshmem-based All-Reduce among CUDA
// GPU-based replicas.
// TODO tixxx consolidate this with
// ALlReduceReduceScatterThunkBase once collective thunk bases
// are consolidated.
class NvshmemAllReduceReduceScatterThunkBase : public NvshmemCollectiveThunk {
 public:
  NvshmemAllReduceReduceScatterThunkBase(
      Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
      std::vector<CollectiveThunk::Buffer> buffers, bool is_sync);

  const CollectiveConfig& config() const override { return config_.config; }
  ReductionKind reduction_kind() const { return config_.reduction_kind; }

  absl::Span<const CollectiveThunk::Buffer> buffers() const { return buffers_; }

 protected:
  const AllReduceConfig config_;
  const std::vector<CollectiveThunk::Buffer> buffers_;
};

// -----------------------------------------------------------------------------
// AllReduce thunk.
// -----------------------------------------------------------------------------

class NvshmemAllReduceStartThunk
    : public NvshmemAllReduceReduceScatterThunkBase {
 public:
  NvshmemAllReduceStartThunk(ThunkInfo thunk_info,
                             const HloAllReduceInstruction* inst,
                             std::vector<CollectiveThunk::Buffer> buffers,
                             bool p2p_memcpy_enabled = false);

  static const char* GetHloOpName() { return "all-reduce-start:nvshmem"; }

  static absl::Status CheckImplementable(const HloAllReduceInstruction* inst,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllReduceInstruction* inst);

 protected:
  absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                    se::Stream& stream) override;
};

// -----------------------------------------------------------------------------

absl::Status RunNvshmemAllReduce(GpuCollectives* collectives,
                                 ReductionKind reduction_kind,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_ALL_REDUCE_THUNK_H_
