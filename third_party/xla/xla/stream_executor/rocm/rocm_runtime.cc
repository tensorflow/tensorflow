/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_runtime.h"

#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/rocm/rocm_driver.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"

#define RETURN_IF_ROCM_ERROR(expr, ...)                             \
  if (auto res = (expr); TF_PREDICT_FALSE(res != hipSuccess)) {     \
    return absl::InternalError(absl::StrCat(                        \
        __VA_ARGS__, ": ", ::stream_executor::gpu::ToString(res))); \
  }

namespace stream_executor {
namespace gpu {

absl::StatusOr<hipFunction_t> RocmRuntime::GetFuncBySymbol(void* symbol) {
  VLOG(2) << "Get ROCM function from a symbol: " << symbol;
#if TF_ROCM_VERSION >= 60200
  hipFunction_t func;
  RETURN_IF_ROCM_ERROR(wrap::hipGetFuncBySymbol(&func, symbol),
                       "Failed call to hipGetFuncBySymbol");
  return func;
#else
  return absl::UnimplementedError("GetFuncBySymbol is not implemented");
#endif  // TF_ROCM_VERSION >= 60200
}

absl::StatusOr<int32_t> RocmRuntime::GetRuntimeVersion() {
  VLOG(2) << "Get ROCM runtime version";
  int32_t version;
  RETURN_IF_ROCM_ERROR(wrap::hipRuntimeGetVersion(&version),
                       "Failed call to hipRuntimeGetVersion");
  return version;
}

}  // namespace gpu

}  // namespace stream_executor
