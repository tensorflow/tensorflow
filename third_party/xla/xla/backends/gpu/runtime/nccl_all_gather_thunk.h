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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NCCL_ALL_GATHER_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NCCL_ALL_GATHER_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/nccl_collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct NcclAllGatherConfig {
  NcclCollectiveConfig config;
};

// Thunk that performs a NCCL-based All-Gather among CUDA GPU-based replicas.
class NcclAllGatherStartThunk : public NcclCollectiveThunk {
 public:
  NcclAllGatherStartThunk(ThunkInfo thunk_info,
                          const HloAllGatherInstruction* inst,
                          std::vector<Buffer> buffers,
                          bool p2p_memcpy_enabled = false);

  static const char* GetHloOpName() { return "all-gather-start"; }

  static absl::Status CheckImplementable(const HloAllGatherInstruction* inst,
                                         int64_t replica_count,
                                         int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllGatherInstruction* inst);

  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 CommunicatorHandle comm_handle) override;

 private:
  bool is_local() const;
  bool should_use_memcpy() const { return p2p_memcpy_enabled_ && is_local(); }

  const NcclAllGatherConfig config_;
  const std::vector<Buffer> buffers_;
  int64_t device_count_ = -1;
  const bool p2p_memcpy_enabled_;

  absl::Mutex pointers_mutex_;
  // Maps from a device to a pointer to an uint64_t array whose size is
  // buffers_.size(). The pointer is written to and read from in each call to
  // RunNcclCollective(), but is preallocated as CUDA host memory in the first
  // call to Initialize(), since allocating CUDA host memory every call to
  // RunNcclCollective() is expensive.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      send_pointers_ ABSL_GUARDED_BY(pointers_mutex_);
  // Maps from a device to a pointer to an uint64_t array whose size is
  // buffers_.size() times the size of the replica_group the device is in. Like
  // with send_pointers, this is used in RunNcclCollective() and allocated in
  // Initialize().
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      receive_pointer_maps_ ABSL_GUARDED_BY(pointers_mutex_);
};

absl::Status RunAllGather(GpuCollectives* collectives,
                          std::vector<DeviceBufferPair>& buffers,
                          se::Stream& stream, Communicator* comm);

absl::Status RunMemCpyAllGather(GpuCollectives* collectives,
                                std::vector<DeviceBufferPair>& buffers,
                                se::Stream& stream, Communicator* comm,
                                RankId rank, uint64_t* send_pointer,
                                uint64_t receive_pointer_map[]);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NCCL_ALL_GATHER_THUNK_H_
