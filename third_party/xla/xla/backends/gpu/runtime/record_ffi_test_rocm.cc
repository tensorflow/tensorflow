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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/rocm/rocm_status.h"  // IWYU pragma: keep
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

absl::StatusOr<void*> LoadNativeFunctionPtr(
    stream_executor::StreamExecutor* executor, absl::Span<const uint8_t> binary,
    absl::string_view name) {
  std::unique_ptr<stream_executor::ActivateContext> activation =
      executor->Activate();
  hipModule_t module = nullptr;
  ABSL_RETURN_IF_ERROR(stream_executor::gpu::ToStatus(
      hipModuleLoadData(&module, binary.data())));
  hipFunction_t function = nullptr;
  std::string kernel_name(name);
  ABSL_RETURN_IF_ERROR(stream_executor::gpu::ToStatus(
      hipModuleGetFunction(&function, module, kernel_name.c_str())));
  return function;
}

}  // namespace xla::gpu
