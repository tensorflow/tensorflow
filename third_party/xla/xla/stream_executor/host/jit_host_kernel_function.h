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

#ifndef XLA_STREAM_EXECUTOR_HOST_JIT_HOST_KERNEL_FUNCTION_H_
#define XLA_STREAM_EXECUTOR_HOST_JIT_HOST_KERNEL_FUNCTION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/host/host_kernel.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"

namespace stream_executor::host {

namespace internal {
class ExecutionEngine;
}

// A host kernel function compiled from LLVM IR at run time
class JitHostKernelFunction : public HostKernel::KernelFunction {
 public:
  SE_HOST_Kernel *kernel() const override { return kernel_; }

  static absl::StatusOr<std::unique_ptr<HostKernel::KernelFunction>>
  CreateFromLlvmIr(absl::string_view name, absl::string_view entry,
                   absl::string_view ir, absl::Span<const std::string> options);

 private:
  explicit JitHostKernelFunction(
      std::unique_ptr<internal::ExecutionEngine> exec_engine);

  std::unique_ptr<internal::ExecutionEngine> engine_;
  SE_HOST_Kernel *kernel_;
};

}  // namespace stream_executor::host

#endif  // XLA_STREAM_EXECUTOR_HOST_JIT_HOST_KERNEL_FUNCTION_H_
