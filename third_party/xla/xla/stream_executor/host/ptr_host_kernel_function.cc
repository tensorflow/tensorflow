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

#include "xla/stream_executor/host/ptr_host_kernel_function.h"

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/host/host_executor.h"
#include "xla/stream_executor/host/host_kernel.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform/initialize.h"

namespace stream_executor::host {

absl::StatusOr<std::unique_ptr<HostKernel::KernelFunction>>
PtrHostKernelFunction::CreateFromPtr(SE_HOST_Kernel *kernel,
                                     absl::string_view kernel_name) {
  return std::unique_ptr<HostKernel::KernelFunction>(
      new PtrHostKernelFunction(kernel));
}

static void RegisterPtrKernelFunctionLoader() {
  using CompiledFunction = std::optional<
      absl::StatusOr<std::unique_ptr<HostKernel::KernelFunction>>>;

  HostExecutor::RegisterKernelFunctionLoader(
      [](const MultiKernelLoaderSpec &spec) -> CompiledFunction {
        if (!spec.has_in_process_symbol()) return std::nullopt;

        return PtrHostKernelFunction::CreateFromPtr(
            reinterpret_cast<SE_HOST_Kernel *>(
                spec.in_process_symbol().symbol()),
            spec.in_process_symbol().kernel_name());
      });
}

}  // namespace stream_executor::host

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    ptr_kernel_function_loader,
    stream_executor::host::RegisterPtrKernelFunctionLoader());
