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

#include "tensorflow/tsl/c/tsl_status_helper.h"

#include "tensorflow/tsl/c/tsl_status_internal.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tsl {

void Set_TSL_Status_from_Status(TSL_Status* tsl_status, const Status& status) {
  absl::StatusCode code = static_cast<absl::StatusCode>(status.code());
  const char* message = tsl::NullTerminatedMessage(status);

  switch (code) {
    case absl::StatusCode::kOk:
      assert(TSL_GetCode(tsl_status) == TSL_OK);
      break;
    case absl::StatusCode::kCancelled:
      TSL_SetStatus(tsl_status, TSL_CANCELLED, message);
      break;
    case absl::StatusCode::kUnknown:
      TSL_SetStatus(tsl_status, TSL_UNKNOWN, message);
      break;
    case absl::StatusCode::kInvalidArgument:
      TSL_SetStatus(tsl_status, TSL_INVALID_ARGUMENT, message);
      break;
    case absl::StatusCode::kDeadlineExceeded:
      TSL_SetStatus(tsl_status, TSL_DEADLINE_EXCEEDED, message);
      break;
    case absl::StatusCode::kNotFound:
      TSL_SetStatus(tsl_status, TSL_NOT_FOUND, message);
      break;
    case absl::StatusCode::kAlreadyExists:
      TSL_SetStatus(tsl_status, TSL_ALREADY_EXISTS, message);
      break;
    case absl::StatusCode::kPermissionDenied:
      TSL_SetStatus(tsl_status, TSL_PERMISSION_DENIED, message);
      break;
    case absl::StatusCode::kUnauthenticated:
      TSL_SetStatus(tsl_status, TSL_UNAUTHENTICATED, message);
      break;
    case absl::StatusCode::kResourceExhausted:
      TSL_SetStatus(tsl_status, TSL_RESOURCE_EXHAUSTED, message);
      break;
    case absl::StatusCode::kFailedPrecondition:
      TSL_SetStatus(tsl_status, TSL_FAILED_PRECONDITION, message);
      break;
    case absl::StatusCode::kAborted:
      TSL_SetStatus(tsl_status, TSL_ABORTED, message);
      break;
    case absl::StatusCode::kOutOfRange:
      TSL_SetStatus(tsl_status, TSL_OUT_OF_RANGE, message);
      break;
    case absl::StatusCode::kUnimplemented:
      TSL_SetStatus(tsl_status, TSL_UNIMPLEMENTED, message);
      break;
    case absl::StatusCode::kInternal:
      TSL_SetStatus(tsl_status, TSL_INTERNAL, message);
      break;
    case absl::StatusCode::kUnavailable:
      TSL_SetStatus(tsl_status, TSL_UNAVAILABLE, message);
      break;
    case absl::StatusCode::kDataLoss:
      TSL_SetStatus(tsl_status, TSL_DATA_LOSS, message);
      break;
    default:
      assert(0);
      break;
  }

  errors::CopyPayloads(status, tsl_status->status);
}

Status StatusFromTSL_Status(const TSL_Status* tsl_status) {
  return tsl_status->status;
}

}  // namespace tsl
