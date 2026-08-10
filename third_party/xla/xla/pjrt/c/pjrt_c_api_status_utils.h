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

#ifndef XLA_PJRT_C_PJRT_C_API_STATUS_UTILS_H_
#define XLA_PJRT_C_PJRT_C_API_STATUS_UTILS_H_

#include <functional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {

// Return error status if not success and frees the PJRT_Error returned by
// `expr`.
#define RETURN_STATUS_IF_PJRT_ERROR(expr, c_api)                         \
  do {                                                                   \
    PJRT_Error* error = (expr);                                          \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(         \
        error, pjrt::MakeErrorDeleter(c_api));                           \
    absl::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                 \
      return _status;                                                    \
    }                                                                    \
  } while (false)

using PJRT_ErrorDeleter = std::function<void(PJRT_Error*)>;

// Pass in an API pointer; receive a custom deleter for smart pointers.
// The lifetime of the Api pointed to must be longer than the error.
PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api);

// Fatal error logging if status is not success. This terminates the process
// and frees the PJRT_Error passed in.
void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api);

absl::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                      const PJRT_Api* api);

PJRT_Error_Code GetErrorCode(const PJRT_Error* error, const PJRT_Api* api);

absl::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api);
absl::Status PjrtErrorToStatus(PJRT_Error* error);
PJRT_Error* StatusToPjRtError(absl::Status s);
void DestroyPjRtError(PJRT_Error* error);

absl::StatusCode PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api);

absl::StatusCode PjrtErrorCodeToStatusCode(PJRT_Error_Code code);
PJRT_Error_Code StatusCodeToPjrtErrorCode(absl::StatusCode code);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_STATUS_UTILS_H_
