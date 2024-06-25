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

#ifndef XLA_SERVICE_CPU_RUNTIME_KERNEL_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_KERNEL_THUNK_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Launches compiled host kernel on the caller thread.
class KernelThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<KernelThunk>> Create(
      Info info, absl::Span<const BufferAllocation::Slice> arguments_buffers,
      absl::Span<const BufferAllocation::Slice> results_buffers,
      std::string kernel_name, se::ThreadDim thread_dim,
      std::optional<int64_t> min_alignment = std::nullopt);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  KernelThunk(Info info,
              absl::Span<const BufferAllocation::Slice> arguments_buffers,
              absl::Span<const BufferAllocation::Slice> results_buffers,
              std::string kernel_name, se::ThreadDim thread_dim,
              std::optional<int64_t> min_alignment);

  std::vector<BufferAllocation::Slice> arguments_buffers_;
  std::vector<BufferAllocation::Slice> results_buffers_;
  std::string kernel_name_;
  se::ThreadDim thread_dim_;
  std::optional<int64_t> min_alignment_;

  // Pointer to the host kernel corresponding to `kernel_name_`. Initialized
  // lazily at run time by looking it up in the HostKernels passed via params.
  //
  // TODO(ezhulenev): This should be moved to initialization stage when we'll
  // have it for CPU thunks.
  std::atomic<SE_HOST_Kernel*> kernel_ptr_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_KERNEL_THUNK_H_
