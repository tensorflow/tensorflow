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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::cpu {

// Launches compiled host kernel on the caller thread.
class KernelThunk final : public Thunk {
 public:
  KernelThunk(Info info, absl::Span<const BufferAllocation::Slice> buffers,
              std::string kernel_name, se::ThreadDim thread_dim);

  absl::Status Execute(const ExecuteParams& params) final;

 private:
  std::vector<BufferAllocation::Slice> buffers_;
  std::string kernel_name_;
  se::ThreadDim thread_dim_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_KERNEL_THUNK_H_
