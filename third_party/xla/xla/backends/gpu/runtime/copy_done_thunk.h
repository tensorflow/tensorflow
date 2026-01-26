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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COPY_DONE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COPY_DONE_THUNK_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

class CopyDoneThunk : public Thunk {
 public:
  CopyDoneThunk(Thunk::Kind kind, ThunkInfo thunk_info,
                std::shared_ptr<CopyThunk::AsyncEvents> events,
                const HloInstruction* copy_start_instr);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const override;

  bool IsAsyncDone() const override { return async_events_ != nullptr; }

 private:
  std::shared_ptr<CopyThunk::AsyncEvents> async_events_;
  const HloInstruction* copy_start_instr_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COPY_DONE_THUNK_H_
