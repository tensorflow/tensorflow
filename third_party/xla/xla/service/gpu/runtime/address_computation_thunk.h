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

#ifndef XLA_SERVICE_GPU_RUNTIME_ADDRESS_COMPUTATION_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_ADDRESS_COMPUTATION_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// AddressComputationThunk wraps the logic to compute dynamic offsets/sizes from
// dynamic-slice or DUS around some original thunks (e.g. custom call or NCCL
// thunks)
//
// AddressComputationThunk assumes that the slices are contiguous.
class AddressComputationThunk : public Thunk {
 public:
  AddressComputationThunk(
      ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
      std::vector<std::optional<const BufferAllocation::Slice>> arguments,
      std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_,
      std::vector<std::optional<std::vector<BufferAllocation::Slice>>>
          offset_buffer_indices,
      std::vector<std::optional<Shape>> orig_shapes,
      std::vector<std::optional<Shape>> sliced_shapes,
      std::vector<std::optional<uint64_t>> offset_byte_sizes);

  AddressComputationThunk(const AddressComputationThunk&) = delete;
  AddressComputationThunk& operator=(const AddressComputationThunk&) = delete;

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::unique_ptr<SequentialThunk> embedded_thunk_;
  std::vector<std::optional<const BufferAllocation::Slice>>
      embedded_thunk_arguments_;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_;
  std::vector<std::optional<std::vector<BufferAllocation::Slice>>>
      offset_buffer_indices_;
  std::vector<std::optional<Shape>> orig_shapes_;
  std::vector<std::optional<Shape>> sliced_shapes_;
  std::vector<std::optional<uint64_t>> offset_byte_sizes_;

  // Pinned host memory for transferring offset values from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      offsets_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_ADDRESS_COMPUTATION_THUNK_H_
