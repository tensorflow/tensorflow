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

#ifndef XLA_SERVICE_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// DynamicSliceThunk wraps the logic to compute dynamic offsets/sizes from
// dynamic-slice or DUS around some original thunks (e.g. custom call or NCCL
// thunks)
//
// DynamicSliceThunk assumes that the slices are contiguous.
class DynamicSliceThunk : public Thunk {
 public:
  struct LoopIter {};

  // Dynamic slice offset can be either: (1) a statically known constant value,
  // (2) a loop iteration number, or (3) a truly dynamic offset that is
  // computed on device and have to be transferred to host.
  using Offset = std::variant<uint64_t, LoopIter, BufferAllocation::Slice>;

  DynamicSliceThunk(
      ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
      std::vector<std::optional<BufferAllocation::Slice>> arguments,
      std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_,
      std::vector<std::optional<std::vector<Offset>>> offsets,
      std::vector<std::optional<Shape>> orig_shapes,
      std::vector<std::optional<Shape>> sliced_shapes,
      std::vector<std::optional<uint64_t>> offset_byte_sizes);

  DynamicSliceThunk(const DynamicSliceThunk&) = delete;
  DynamicSliceThunk& operator=(const DynamicSliceThunk&) = delete;

  const Thunk* embedded_thunk() const { return embedded_thunk_.get(); }

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::unique_ptr<SequentialThunk> embedded_thunk_;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_;

  // Definition of a dynamic slice that extract a slice from the original buffer
  // defined by `embedded_thunk_argument` at given `offsets`.
  struct SliceDef {
    std::optional<BufferAllocation::Slice> embedded_thunk_argument;
    std::optional<std::vector<Offset>> offsets;
    std::optional<Shape> orig_shape;
    std::optional<Shape> sliced_shape;
    std::optional<uint64_t> offset_byte_size;
  };

  std::vector<SliceDef> slices_;

  // Pinned host memory for transferring offset values from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      offsets_allocs_ ABSL_GUARDED_BY(mutex_);

  // Pre-computed size requirement for `offsets_allocs_`.
  int64_t offsets_allocs_size_ = 0;

  // A mapping from argument index to the base offset in the `offsets_allocs_`.
  std::vector<int64_t> offsets_allocs_base_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_
