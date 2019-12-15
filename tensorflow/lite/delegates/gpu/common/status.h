/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_STATUS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_STATUS_H_

#include <string>

namespace tflite {
namespace gpu {

enum class StatusCode {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
  kDoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

// Lite version of Status without dependency on protobuf.
// TODO(b/128867901): Migrate to absl::Status.
class Status {
 public:
  Status() = default;
  Status(StatusCode code) : code_(code) {}
  Status(StatusCode code, const std::string& error_message)
      : code_(code), error_message_(error_message) {}

  const std::string& error_message() const { return error_message_; }
  StatusCode code() const { return code_; }
  bool ok() const { return code_ == StatusCode::kOk; }

  void IgnoreError() const {}

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string error_message_;
};

#define RETURN_IF_ERROR(status)        \
  {                                    \
    const auto status2 = (status);     \
    if (!status2.ok()) return status2; \
  }

inline Status OkStatus() { return Status(); }

inline Status AlreadyExistsError(const std::string& message) {
  return Status(StatusCode::kAlreadyExists, message);
}

inline Status DeadlineExceededError(const std::string& message) {
  return Status(StatusCode::kDeadlineExceeded, message);
}

inline Status FailedPreconditionError(const std::string& message) {
  return Status(StatusCode::kFailedPrecondition, message);
}

inline Status InternalError(const std::string& message) {
  return Status(StatusCode::kInternal, message);
}

inline Status InvalidArgumentError(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

inline Status NotFoundError(const std::string& message) {
  return Status(StatusCode::kNotFound, message);
}

inline Status OutOfRangeError(const std::string& message) {
  return Status(StatusCode::kOutOfRange, message);
}

inline Status PermissionDeniedError(const std::string& message) {
  return Status(StatusCode::kPermissionDenied, message);
}

inline Status ResourceExhaustedError(const std::string& message) {
  return Status(StatusCode::kResourceExhausted, message);
}

inline Status UnavailableError(const std::string& message) {
  return Status(StatusCode::kUnavailable, message);
}

inline Status UnimplementedError(const std::string& message) {
  return Status(StatusCode::kUnimplemented, message);
}

inline Status UnknownError(const std::string& message) {
  return Status(StatusCode::kUnknown, message);
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_STATUS_H_
