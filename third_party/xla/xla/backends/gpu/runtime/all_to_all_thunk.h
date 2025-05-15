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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_

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
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct AllToAllConfig {
  CollectiveConfig config;
  bool has_split_dimension;
};

// Thunk that performs an All-to-All among CUDA GPU-based replicas.
class AllToAllStartThunk : public CollectiveThunk {
 public:
  AllToAllStartThunk(ThunkInfo thunk_info, const HloAllToAllInstruction* instr,
                     std::vector<Buffer> buffers, bool p2p_memcpy_enabled);

  // Returns whether the given instruction can be lowered to an all-to-all
  // call.
  static absl::Status CheckImplementable(const HloAllToAllInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllToAllInstruction* instr);

  const CollectiveConfig& config() const override { return config_.config; }
  bool has_split_dimension() const { return config_.has_split_dimension; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm) override;

  AsyncStreamKind GetAsyncStreamKind() const override;

  bool is_local() const;

 private:
  const AllToAllConfig config_;
  const std::vector<Buffer> buffers_;
  int64_t device_count_ = 1;
  bool p2p_memcpy_enabled_ = false;
  absl::Mutex pointer_maps_mutex_;
  // Maps from a device to a uint64_t array of size num_devices. The array is
  // written to and used in each call to RunCollective(), but is
  // preallocated as CUDA host memory in the first call to Initialize(), since
  // allocating CUDA host memory every call to RunCollective() is expensive.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      send_pointer_maps_ ABSL_GUARDED_BY(pointer_maps_mutex_);
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      receive_pointer_maps_ ABSL_GUARDED_BY(pointer_maps_mutex_);
};

absl::Status RunAllToAll(GpuCollectives* collectives, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, Communicator* comm);

absl::Status RunMemCpyAllToAll(GpuCollectives* collectives,
                               bool has_split_dimension,
                               std::vector<DeviceBufferPair>& buffers,
                               se::Stream& stream, Communicator* comm,
                               uint64_t send_pointer_map[],
                               uint64_t receive_pointer_map[]);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_
