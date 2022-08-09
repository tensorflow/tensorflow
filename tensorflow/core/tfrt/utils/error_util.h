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

#include "tensorflow/core/platform/status.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
class AsyncValue;
class DecodedDiagnostic;

tfrt::ErrorCode ConvertTfErrorCodeToTfrtErrorCode(
    const tensorflow::Status& status);

tensorflow::Status CreateTfErrorStatus(const DecodedDiagnostic& error);

tensorflow::Status ToTfStatus(const AsyncValue* av);

inline std::string MakeStatusString(tensorflow::Status status) {
  switch (status.code()) {
    case tensorflow::error::OK:
      return "OK";
    case tensorflow::error::CANCELLED:
      return absl::StrCat("Cancelled: ", status.error_message());
    case tensorflow::error::UNKNOWN:
      return absl::StrCat("Unknown: ", status.error_message());
    case tensorflow::error::INVALID_ARGUMENT:
      return absl::StrCat("Invalid argument: ", status.error_message());
    case tensorflow::error::DEADLINE_EXCEEDED:
      return absl::StrCat("Deadline exceeded: ", status.error_message());
    case tensorflow::error::NOT_FOUND:
      return absl::StrCat("Not found: ", status.error_message());
    case tensorflow::error::ALREADY_EXISTS:
      return absl::StrCat("Already exists: ", status.error_message());
    case tensorflow::error::PERMISSION_DENIED:
      return absl::StrCat("Permission denied: ", status.error_message());
    case tensorflow::error::UNAUTHENTICATED:
      return absl::StrCat("Unauthenticated: ", status.error_message());
    case tensorflow::error::RESOURCE_EXHAUSTED:
      return absl::StrCat("Resource exhausted: ", status.error_message());
    case tensorflow::error::FAILED_PRECONDITION:
      return absl::StrCat("Failed precondition: ", status.error_message());
    case tensorflow::error::ABORTED:
      return absl::StrCat("Aborted: ", status.error_message());
    case tensorflow::error::OUT_OF_RANGE:
      return absl::StrCat("Out of range: ", status.error_message());
    case tensorflow::error::UNIMPLEMENTED:
      return absl::StrCat("Unimplemented: ", status.error_message());
    case tensorflow::error::INTERNAL:
      return absl::StrCat("Internal: ", status.error_message());
    case tensorflow::error::UNAVAILABLE:
      return absl::StrCat("Unavailable: ", status.error_message());
    case tensorflow::error::DATA_LOSS:
      return absl::StrCat("Data loss: ", status.error_message());
    default:
      return absl::StrCat("Unknown code: ", status.error_message());
  }
}

inline llvm::Error MakeStatusError(tensorflow::Status status) {
  return MakeStringError(MakeStatusString(status));
}

tensorflow::Status TfStatusFromAbslStatus(absl::Status status);

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_ERROR_UTIL_H_
