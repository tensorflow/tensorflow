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

#ifndef XLA_BACKENDS_CPU_RUNTIME_TOPK_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_TOPK_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

class TopKThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<TopKThunk>> Create(
      Info info, BufferAllocation::Slice values, BufferAllocation::Slice output,
      BufferAllocation::Slice indices, int64_t batch_size, int64_t input_size,
      int64_t k);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final {
    return {BufferUse::Read(values_buffer_), BufferUse::Write(output_buffer_),
            BufferUse::Write(indices_buffer_)};
  }

 private:
  TopKThunk(Info info, BufferAllocation::Slice values,
            BufferAllocation::Slice output, BufferAllocation::Slice indices,
            int64_t batch_size, int64_t input_size, int64_t k);

  BufferAllocation::Slice values_buffer_;
  BufferAllocation::Slice output_buffer_;
  BufferAllocation::Slice indices_buffer_;
  int64_t batch_size_;
  int64_t input_size_;
  int64_t k_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_TOPK_THUNK_H_
