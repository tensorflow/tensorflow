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

#include "xla/service/gpu/nccl_api.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/nccl/nccl.h"
#include "xla/service/gpu/nccl_clique_key.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NCCL errors.
//==-----------------------------------------------------------------------===//

static absl::Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                             const char* expr) {
  if (s == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "%s:%d: NCCL operation %s failed: %s."
      " Last NCCL warning(error) log entry (may be unrelated) '%s'.",
      file, line, expr, ncclGetErrorString(s), ncclGetLastError(nullptr)));
}

#define XLA_NCCL_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NCCL_RETURN_IF_ERROR(expr)      \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      return s;                             \
    }                                       \
  } while (0)

#define XLA_NCCL_LOG_IF_ERROR(expr)         \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      LOG(ERROR) << s.ToString();           \
    }                                       \
  } while (0)

//==-----------------------------------------------------------------------===//
// NcclApi
//==-----------------------------------------------------------------------===//

absl::StatusOr<NcclCliqueId> NcclApi::GetUniqueId() {
  static_assert(NCCL_UNIQUE_ID_BYTES == NcclCliqueId::kSize,
                "size of nccl unique id must match the clique id size");
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return NcclCliqueId(id.internal);
}

}  // namespace xla::gpu
