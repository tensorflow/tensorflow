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

#include <cstdint>

#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"

namespace xla::cpu {

// Copies data from a source buffer to a destination buffer.
class CopyThunk final : public Thunk {
 public:
  CopyThunk(BufferAllocation::Slice source_buffer,
            BufferAllocation::Slice destination_buffer, uint64_t size_in_bytes);

  absl::Status Execute(const ExecuteParams& params) final;

 private:
  BufferAllocation::Slice source_buffer_;
  BufferAllocation::Slice destination_buffer_;
  uint64_t size_in_bytes_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_COPY_THUNK_H_
