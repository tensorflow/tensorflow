/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_REDUCE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_REDUCE_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Per-`StreamExecutor` state used only when the reduce root is selected at run
// time. `reduce_roots` is a host-pinned staging buffer that the trailing S32
// root operand is copied into before the reduce is launched.
struct CollectiveReduceMetadata {
  int64_t num_roots = 0;
  std::unique_ptr<se::MemoryAllocation> reduce_roots = nullptr;
};

// Thunk that performs a NCCL-based reduce (`ncclReduce`): the operands are
// reduced with the `to_apply` computation and the result is written only to the
// root rank. The root is rank 0 within the replica group by default, or chosen
// at run time from a trailing S32 operand when `has_dynamic_root` is set.
class CollectiveReduceThunk : public AllReduceReduceScatterThunkBase {
 public:
  CollectiveReduceThunk(ThunkInfo thunk_info, AllReduceConfig config,
                        std::vector<Buffer> buffers,
                        bool has_dynamic_root = false);
  CollectiveReduceThunk(ThunkInfo thunk_info,
                        const HloCollectiveReduceInstruction* inst,
                        std::vector<Buffer> buffers,
                        bool p2p_memcpy_enabled = false,
                        bool has_dynamic_root = false);

  static absl::string_view GetHloOpName() { return "collective-reduce-start"; }

  static absl::Status CheckImplementable(
      const HloCollectiveReduceInstruction* inst, int64_t replica_count,
      int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectiveReduceInstruction* inst);

  static absl::StatusOr<std::unique_ptr<CollectiveReduceThunk>> FromProto(
      ThunkInfo thunk_info, const CollectiveReduceThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 protected:
  bool RequiresRendezvous() const override { return true; }

  absl::Status InitializeCollective(const InitializeParams& params,
                                    const GpuCliqueKey& clique_key) override;

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

 private:
  const bool has_dynamic_root_;
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<CollectiveReduceMetadata>>
      per_executor_metadata_ ABSL_GUARDED_BY(mutex_);
};

// Runs a reduce over `buffers` on `stream`. When `has_dynamic_root` is set the
// last entry of `buffers` is the S32 root vector (not itself reduced) and
// `metadata` provides the host staging buffer for the root ranks.
absl::Status RunCollectiveReduce(ReductionKind reduction_kind,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream, Communicator& comm,
                                 CollectiveReduceMetadata* metadata = nullptr,
                                 bool has_dynamic_root = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_REDUCE_THUNK_H_
