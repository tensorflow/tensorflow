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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct AllReduceConfig {
  CollectiveConfig config;
  ReductionKind reduction_kind;
};

AllReduceConfig GetAllReduceConfigInst(const HloAllReduceInstructionBase* inst);

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class AllReduceReduceScatterThunkBase : public CollectiveThunk {
 public:
  AllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                  AllReduceConfig config,
                                  std::vector<Buffer> buffers, bool is_sync);
  AllReduceReduceScatterThunkBase(
      Kind kind, ThunkInfo thunk_info, AllReduceConfig config,
      std::vector<Buffer> buffers,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  const CollectiveConfig& config() const override { return config_.config; }
  ReductionKind reduction_kind() const { return config_.reduction_kind; }

  absl::Span<const Buffer> buffers() const { return buffers_; }

  BufferUses buffer_uses() const override {
    BufferUses uses;
    uses.reserve(buffers_.size() * 2);
    for (const Buffer& buffer : buffers_) {
      uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                     buffer.source_buffer.shape));
      uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                      buffer.destination_buffer.shape));
    }
    return uses;
  }

 protected:
  const AllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

// -----------------------------------------------------------------------------
// AllReduce thunk.
// -----------------------------------------------------------------------------

class AllReduceStartThunk : public AllReduceReduceScatterThunkBase {
 public:
  AllReduceStartThunk(
      ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
      std::vector<Buffer> buffers,
      std::unique_ptr<CollectiveKernelThunk> collective_kernel_thunk,
      bool p2p_memcpy_enabled = false);
  AllReduceStartThunk(
      ThunkInfo thunk_info, const AllReduceConfig& config,
      std::vector<Buffer> buffers,
      std::unique_ptr<CollectiveKernelThunk> collective_kernel_thunk,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  static absl::string_view GetHloOpName() { return "all-reduce-start"; }

  static absl::Status CheckImplementable(const HloAllReduceInstruction* inst,
                                         int64_t replica_count,
                                         int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllReduceInstruction* inst);

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;

  static absl::StatusOr<std::unique_ptr<AllReduceStartThunk>> FromProto(
      ThunkInfo thunk_info, const AllReduceStartThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CollectiveThunk::AsyncEventsMap& async_events_map);

  absl::StatusOr<ThunkProto> ToProto() const override;

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;

 private:
  std::unique_ptr<CollectiveKernelThunk> collective_kernel_thunk_;
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

  static absl::string_view GetHloOpName() { return "reduce-scatter-start"; }

  static absl::Status CheckImplementable(
      const HloReduceScatterInstruction* inst, int64_t replica_count,
      int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloReduceScatterInstruction* inst);

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;
};

// -----------------------------------------------------------------------------

absl::Status RunAllReduce(ReductionKind reduction_kind,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator& comm,
                          bool use_symmetric_buffer = false);

absl::Status RunReduceScatter(ReductionKind reduction_kind,
                              std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, Communicator& comm,
                              bool use_symmetric_buffer = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_THUNK_H_
