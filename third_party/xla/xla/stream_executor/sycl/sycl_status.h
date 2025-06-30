/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_

#include <string>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

enum class SyclError {
  kSyclSuccess,
  kSyclErrorNoDevice,
  kSyclErrorNotReady,
  kSyclErrorInvalidDevice,
  kSyclErrorInvalidPointer,
  kSyclErrorInvalidStream,
  kSyclErrorDestroyDefaultStream,
  kSyclErrorZeError,
};

// Returns a textual description of the given SyclError.
std::string ToString(SyclError error);

// Returns an absl::Status corresponding to the SyclError.
inline absl::Status ToStatus(SyclError result, absl::string_view detail = "") {
  if (ABSL_PREDICT_TRUE(result == SyclError::kSyclSuccess)) {
    return absl::OkStatus();
  }
  std::string error_message = absl::StrCat(detail, ": ", ToString(result));
  return absl::InternalError(error_message);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_STATUS_H_
