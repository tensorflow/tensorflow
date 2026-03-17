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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

class CopyThunk : public Thunk {
 public:
  class AsyncEvents {
   public:
    // Add a new copy-start completion event.
    absl::Status Emplace(se::StreamExecutor* executor, int64_t instr_id,
                         std::unique_ptr<se::Event> event);

    // Retrieve a completion event started by copy-start instruction
    // `instr`, and remove the event from the collection.
    absl::StatusOr<std::unique_ptr<se::Event>> Extract(
        se::StreamExecutor* executor, int64_t instr_id);

   private:
    using Key = std::pair<se::StreamExecutor*, int64_t>;
    absl::Mutex mutex_;
    absl::flat_hash_map<Key, std::unique_ptr<se::Event>> events_
        ABSL_GUARDED_BY(mutex_);
  };

  using AsyncEventsMap =
      absl::flat_hash_map<AsyncEventsUniqueId, std::shared_ptr<AsyncEvents>>;

  CopyThunk(ThunkInfo thunk_info, const ShapedSlice& source_buffer,
            const ShapedSlice& destination_buffer, int64_t mem_size);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const ShapedSlice& source() const { return source_buffer_; }
  const ShapedSlice& destination() const { return destination_buffer_; }
  uint64_t size_bytes() const { return mem_size_; }

  BufferUses buffer_uses() const override {
    return {
        BufferUse::Read(source_buffer_.slice, source_buffer_.shape),
        BufferUse::Write(destination_buffer_.slice, destination_buffer_.shape),
    };
  }

  bool operator==(const CopyThunk& other) const {
    return source() == other.source() && destination() == other.destination() &&
           size_bytes() == other.size_bytes();
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<CopyThunk>> FromProto(
      ThunkInfo thunk_info, const CopyThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

 private:
  const ShapedSlice source_buffer_;
  const ShapedSlice destination_buffer_;
  const int64_t mem_size_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COPY_THUNK_H_
