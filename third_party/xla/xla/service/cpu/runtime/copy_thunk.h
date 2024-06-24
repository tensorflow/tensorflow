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

#ifndef XLA_SERVICE_CPU_RUNTIME_COPY_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_COPY_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/transpose.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Copies data from a source buffer to a destination buffer. If source and
// destination buffers have different layouts it will transpose the data.
class CopyThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<CopyThunk>> Create(
      Info info, BufferAllocation::Slice source_buffer,
      const Shape& source_shape, BufferAllocation::Slice destination_buffer,
      const Shape& destination_shape);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final {
    return {{source_buffer_, BufferUse::kRead},
            {destination_buffer_, BufferUse::kWrite}};
  }

 private:
  CopyThunk(Info info, BufferAllocation::Slice source_buffer,
            const Shape& source_shape,
            BufferAllocation::Slice destination_buffer,
            const Shape& destination_shape);

  BufferAllocation::Slice source_buffer_;
  Shape source_shape_;

  BufferAllocation::Slice destination_buffer_;
  Shape destination_shape_;

  std::unique_ptr<TransposePlan> transpose_plan_;  // optional
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_COPY_THUNK_H_
