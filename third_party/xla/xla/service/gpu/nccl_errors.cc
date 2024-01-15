/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_errors.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/service/gpu/nccl_types.h"

namespace xla::gpu {

absl::Status ToStatus(NcclStatus s, const char* file, int64_t line,
                      const char* expr) {
#ifdef XLA_ENABLE_XCCL
  if (s == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "%s:%d: NCCL operation %s failed: %s."
      " Last NCCL warning(error) log entry (may be unrelated) '%s'.",
      file, line, expr, ncclGetErrorString(s), ncclGetLastError(nullptr)));
#else
  return absl::InternalError("XLA compiled without NCCL support");
#endif
}

}  // namespace xla::gpu
