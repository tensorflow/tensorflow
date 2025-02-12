/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NCCL_RAGGED_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NCCL_RAGGED_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/nccl_collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct NcclRaggedAllToAllConfig {
  NcclCollectiveConfig config;
  int64_t num_total_updates = 1;
  int64_t ragged_row_element_size = 1;
};

// Thunk that performs a NCCL-based Ragged-All-to-All among CUDA GPU-based
// replicas.
class NcclRaggedAllToAllStartThunk : public NcclCollectiveThunk {
 public:
  NcclRaggedAllToAllStartThunk(ThunkInfo thunk_info,
                               const HloRaggedAllToAllInstruction* instr,
                               std::vector<Buffer> buffers,
                               bool p2p_memcpy_enabled);

  // Returns whether the given instruction can be lowered to a nccl
  // ragged-all-to-all call.
  static absl::Status CheckImplementable(
      const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "ragged-all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloRaggedAllToAllInstruction* instr);

  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 CommunicatorHandle comm_handle) override;

  AsyncStreamKind GetAsyncStreamKind() const override;

 private:
  bool is_local() const;
  bool should_use_memcpy() const { return p2p_memcpy_enabled_ && is_local(); }

  const NcclRaggedAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
  int64_t device_count_ = -1;
  const bool p2p_memcpy_enabled_;

  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::vector<std::unique_ptr<se::MemoryAllocation>>>
      host_buffer_allocs_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_map<se::StreamExecutor*, se::DeviceMemoryHandle>
      device_buffer_allocs_ ABSL_GUARDED_BY(mutex_);

  absl::Mutex pointers_mutex_;
  // Maps from a device to a pointer to an uint64_t value. The pointer is
  // written to and read from in each call to RunNcclCollective(), but is
  // preallocated as CUDA host memory in the first call to Initialize(), since
  // allocating CUDA host memory every call to RunNcclCollective() is expensive.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      send_pointers_ ABSL_GUARDED_BY(pointers_mutex_);
  // Maps from a device to a pointer to an uint64_t array whose size is the
  // size of the replica_group the device is in. Like with send_pointers,
  // this is used in RunNcclCollective() and allocated in Initialize().
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      receive_pointer_maps_ ABSL_GUARDED_BY(pointers_mutex_);
};

absl::Status RunRaggedAllToAll(
    GpuCollectives* collectives, int64_t ragged_row_element_size,
    int64_t num_total_updates, const std::vector<DeviceBufferPair>& buffers,
    se::Stream& stream, Communicator* comm,
    const std::vector<int64_t*>& ragged_metadata_allocs,
    const se::DeviceMemoryBase& output_offsets_device_buffer);

absl::Status RunMemCpyRaggedAllToAll(
    GpuCollectives* collectives, int64_t ragged_row_element_size,
    int64_t num_total_updates, const std::vector<DeviceBufferPair>& buffers,
    se::Stream& stream, Communicator* comm,
    const std::vector<int64_t*>& ragged_metadata_allocs, uint64_t* send_pointer,
    uint64_t receive_pointer_map[]);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NCCL_RAGGED_ALL_TO_ALL_THUNK_H_
