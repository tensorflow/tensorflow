/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_ERROR_UTIL_H_
#define TENSORFLOW_CORE_TFRT_UTILS_ERROR_UTIL_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
class DecodedDiagnostic;

tfrt::ErrorCode ConvertTfErrorCodeToTfrtErrorCode(const absl::Status& status);

absl::Status CreateTfErrorStatus(const DecodedDiagnostic& error);

absl::Status ToTfStatus(const AsyncValue* av);

inline std::string MakeStatusString(absl::Status status) {
  switch (static_cast<absl::StatusCode>(status.code())) {
    case absl::StatusCode::kOk:
      return "OK";
    case absl::StatusCode::kCancelled:
      return absl::StrCat("Cancelled: ", status.message());
    case absl::StatusCode::kUnknown:
      return absl::StrCat("Unknown: ", status.message());
    case absl::StatusCode::kInvalidArgument:
      return absl::StrCat("Invalid argument: ", status.message());
    case absl::StatusCode::kDeadlineExceeded:
      return absl::StrCat("Deadline exceeded: ", status.message());
    case absl::StatusCode::kNotFound:
      return absl::StrCat("Not found: ", status.message());
    case absl::StatusCode::kAlreadyExists:
      return absl::StrCat("Already exists: ", status.message());
    case absl::StatusCode::kPermissionDenied:
      return absl::StrCat("Permission denied: ", status.message());
    case absl::StatusCode::kUnauthenticated:
      return absl::StrCat("Unauthenticated: ", status.message());
    case absl::StatusCode::kResourceExhausted:
      return absl::StrCat("Resource exhausted: ", status.message());
    case absl::StatusCode::kFailedPrecondition:
      return absl::StrCat("Failed precondition: ", status.message());
    case absl::StatusCode::kAborted:
      return absl::StrCat("Aborted: ", status.message());
    case absl::StatusCode::kOutOfRange:
      return absl::StrCat("Out of range: ", status.message());
    case absl::StatusCode::kUnimplemented:
      return absl::StrCat("Unimplemented: ", status.message());
    case absl::StatusCode::kInternal:
      return absl::StrCat("Internal: ", status.message());
    case absl::StatusCode::kUnavailable:
      return absl::StrCat("Unavailable: ", status.message());
    case absl::StatusCode::kDataLoss:
      return absl::StrCat("Data loss: ", status.message());
    default:
      return absl::StrCat("Unknown code: ", status.message());
  }
}

inline llvm::Error MakeStatusError(absl::Status status) {
  return MakeStringError(MakeStatusString(status));
}

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_ERROR_UTIL_H_
