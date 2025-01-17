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

#ifndef XLA_BACKENDS_CPU_RUNTIME_COPY_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_COPY_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/pjrt/transpose.h"
#include "xla/runtime/buffer_use.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Copies data from a source buffer to a destination buffer. If source and
// destination buffers have different layouts it will transpose the data.
class CopyThunk final : public Thunk {
 public:
  // Parameters for running a copy operation in parallel.
  struct ParallelBlockParams {
    int64_t size_in_bytes;
    int64_t block_size;
    int64_t block_count;
  };

  static absl::StatusOr<std::unique_ptr<CopyThunk>> Create(
      Info info, BufferAllocation::Slice src_buffer, const Shape& src_shape,
      BufferAllocation::Slice dst_buffer, const Shape& dst_shape);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final {
    return {{src_buffer_, BufferUse::kRead}, {dst_buffer_, BufferUse::kWrite}};
  }

  const Shape& src_shape() const { return src_shape_; }
  const Shape& dst_shape() const { return dst_shape_; }

  const BufferAllocation::Slice& src_buffer() const { return src_buffer_; }
  const BufferAllocation::Slice& dst_buffer() const { return dst_buffer_; }

 private:
  CopyThunk(Info info, BufferAllocation::Slice src_buffer,
            const Shape& src_shape, BufferAllocation::Slice dst_buffer,
            const Shape& dst_shape);

  static ParallelBlockParams ComputeParallelBlockParams(const Shape& shape);

  BufferAllocation::Slice src_buffer_;
  Shape src_shape_;

  BufferAllocation::Slice dst_buffer_;
  Shape dst_shape_;

  ParallelBlockParams parallel_block_params_;
  std::unique_ptr<TransposePlan> transpose_plan_;  // optional
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_COPY_THUNK_H_
