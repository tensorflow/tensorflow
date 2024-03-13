/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/c/tsl_status_helper.h"

#include "xla/tsl/c/tsl_status_internal.h"
#include "tsl/platform/errors.h"

namespace tsl {

TSL_Code TSLCodeFromStatusCode(absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kOk:
      return TSL_OK;
    case absl::StatusCode::kCancelled:
      return TSL_CANCELLED;
    case absl::StatusCode::kInvalidArgument:
      return TSL_INVALID_ARGUMENT;
    case absl::StatusCode::kDeadlineExceeded:
      return TSL_DEADLINE_EXCEEDED;
    case absl::StatusCode::kNotFound:
      return TSL_NOT_FOUND;
    case absl::StatusCode::kAlreadyExists:
      return TSL_ALREADY_EXISTS;
    case absl::StatusCode::kPermissionDenied:
      return TSL_PERMISSION_DENIED;
    case absl::StatusCode::kUnauthenticated:
      return TSL_UNAUTHENTICATED;
    case absl::StatusCode::kResourceExhausted:
      return TSL_RESOURCE_EXHAUSTED;
    case absl::StatusCode::kFailedPrecondition:
      return TSL_FAILED_PRECONDITION;
    case absl::StatusCode::kAborted:
      return TSL_ABORTED;
    case absl::StatusCode::kOutOfRange:
      return TSL_OUT_OF_RANGE;
    case absl::StatusCode::kUnimplemented:
      return TSL_UNIMPLEMENTED;
    case absl::StatusCode::kInternal:
      return TSL_INTERNAL;
    case absl::StatusCode::kUnavailable:
      return TSL_UNAVAILABLE;
    case absl::StatusCode::kDataLoss:
      return TSL_DATA_LOSS;
    default:
      return TSL_UNKNOWN;
  }
}

absl::StatusCode StatusCodeFromTSLCode(TSL_Code code) {
  switch (code) {
    case TSL_OK:
      return absl::StatusCode::kOk;
    case TSL_CANCELLED:
      return absl::StatusCode::kCancelled;
    case TSL_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case TSL_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case TSL_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case TSL_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case TSL_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case TSL_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
    case TSL_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case TSL_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case TSL_ABORTED:
      return absl::StatusCode::kAborted;
    case TSL_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case TSL_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case TSL_INTERNAL:
      return absl::StatusCode::kInternal;
    case TSL_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case TSL_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    default:
      return absl::StatusCode::kUnknown;
  }
}

}  // namespace tsl
