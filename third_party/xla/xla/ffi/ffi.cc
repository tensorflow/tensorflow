/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/ffi/ffi.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "tsl/platform/logging.h"

//===----------------------------------------------------------------------===//
// XLA FFI C structs definition
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error {
  absl::Status status;
};

//===----------------------------------------------------------------------===//

namespace xla::ffi {

absl::Status Unwrap(XLA_FFI_Error* error) {
  if (error == nullptr) return absl::OkStatus();
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

//===----------------------------------------------------------------------===//
// XLA FFI Api Implementation
//===----------------------------------------------------------------------===//

static std::string StructSizeErrorMsg(std::string_view struct_name,
                                      size_t expected_size,
                                      size_t actual_size) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ",
                      expected_size, ", got ", actual_size,
                      ". Check installed software versions. ",
                      "The framework XLA FFI API version is ",
                      XLA_FFI_API_MAJOR, ".", XLA_FFI_API_MINOR, ".");
}

static absl::Status ActualStructSizeIsGreaterOrEqual(
    std::string_view struct_name, size_t expected_size, size_t actual_size) {
  if (actual_size < expected_size) {
    return absl::InvalidArgumentError(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  if (actual_size > expected_size) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected_size, actual_size);
  }
  return absl::OkStatus();
}

static absl::StatusCode ToStatusCode(XLA_FFI_Error_Code errc) {
  switch (errc) {
    case XLA_FFI_Error_Code_CANCELLED:
      return absl::StatusCode::kCancelled;
    case XLA_FFI_Error_Code_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case XLA_FFI_Error_Code_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case XLA_FFI_Error_Code_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case XLA_FFI_Error_Code_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case XLA_FFI_Error_Code_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case XLA_FFI_Error_Code_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case XLA_FFI_Error_Code_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case XLA_FFI_Error_Code_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case XLA_FFI_Error_Code_ABORTED:
      return absl::StatusCode::kAborted;
    case XLA_FFI_Error_Code_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case XLA_FFI_Error_Code_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case XLA_FFI_Error_Code_INTERNAL:
      return absl::StatusCode::kInternal;
    case XLA_FFI_Error_Code_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case XLA_FFI_Error_Code_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case XLA_FFI_Error_Code_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
  }
}

#define XLA_FFI_RETURN_IF_ERROR(expr)                                   \
  do {                                                                  \
    absl::Status _status = (expr);                                      \
    if (!_status.ok()) {                                                \
      XLA_FFI_Error* _c_status = new XLA_FFI_Error{std::move(_status)}; \
      return _c_status;                                                 \
    }                                                                   \
  } while (false)

static XLA_FFI_Error* XLA_FFI_Error_Create(XLA_FFI_Error_Create_Args* args) {
  XLA_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "XLA_FFI_Error_Create", XLA_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size));

  return new XLA_FFI_Error{
      absl::Status(ToStatusCode(args->errc), args->message)};
}

static XLA_FFI_Error* XLA_FFI_Error_Forward(void* status) {
  return new XLA_FFI_Error{std::move(*reinterpret_cast<absl::Status*>(status))};
}

//===----------------------------------------------------------------------===//
// XLA FFI Api access
//===----------------------------------------------------------------------===//

static XLA_FFI_InternalApi internal_api = {
    XLA_FFI_Error_Forward,
};

static XLA_FFI_Api api = {
    XLA_FFI_Api_STRUCT_SIZE,
    /*priv=*/nullptr,

    &internal_api,

    // Error reporting API
    XLA_FFI_Error_Create,
};

XLA_FFI_Api* GetXlaFfiApi() { return &api; }

}  // namespace xla::ffi
