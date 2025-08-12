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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_ERRORS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_ERRORS_H_

#include <atomic>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"  // IWYU pragma: keep
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"  // IWYU pragma: keep
                                                       //
#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

//===----------------------------------------------------------------------===//
// Collection of helper macros for handling NCCL errors.
//===----------------------------------------------------------------------===//

#define XLA_NCCL_STATUS(expr)                                         \
  [](ncclResult_t s, absl::string_view str) -> absl::Status {         \
    if (s == ncclSuccess || s == ncclInProgress) {                    \
      return absl::OkStatus();                                        \
    }                                                                 \
    return xla::Internal(                                             \
        "NCCL operation %s failed: %s. Last NCCL warning(error) log " \
        "entry (may be unrelated) '%s'.",                             \
        str, ncclGetErrorString(s), ncclGetLastError(nullptr));       \
  }(expr, #expr)

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

#define XLA_NCCL_CHECK(expr) CHECK(XLA_NCCL_STATUS(expr).ok())

namespace xla::gpu {

// Polls the provided communicator until it is "done" or aborted.
//
// NCCL communicators can be blocking or non-blocking. Operations performed on
// non-blocking communicators return immediately, and it is the responsibility
// of the programmer to repeatedly call ncclCommGetAsyncError on the
// communicator until ncclCommGetAsyncError no long returns inProgress. That is
// what PollUntilDone does.
//
// Note, however, that the semantics of NCCL collectives are a bit subtle. For
// example, a collective operation may report itself as done when it is
// scheduled on the GPU but has not yet executed. Refer to the NCCL
// documentation and exercise caution when reasoning about whether an operation
// is really "done".
absl::Status PollUntilDone(ncclComm_t comm, const std::atomic_bool& aborted);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_ERRORS_H_
