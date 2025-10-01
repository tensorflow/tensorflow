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

#ifndef XLA_ERROR_ERROR_CODES_H_
#define XLA_ERROR_ERROR_CODES_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/error/debug_me_context_util.h"

namespace xla::error {

// The single source of truth for all XLA error codes. These error codes provide
// a consistent way to report errors across XLA. They provide finer grained
// error codes than absl::Status codes and each error code is linked to a
// corresponding page in the XLA documentation with more details about the error
// and potential solutions.

// The format of the macro declaration is:
// X(ErrorCode, ErrorCodeName, absl::StatusCode)
// where:
// - ErrorCode is the canonical identifier of the error code - all XLA error
//   messages are prefixed with this identifier. Additionally all XLA
//   documentation is indexed by this identifier.
// - ErrorCodeName is a canonical name for the error code also present in all
//   XLA error messsages. Additionally used in the XLA codebase as the ErrorCode
//   Enum value.
// - absl::StatusCode is the canonical absl::StatusCode that most closely
//   matches the error code.
// e.g. For X("E1234", BinaryTooLarge, absl::StatusCode::kResourceExhausted)
// we generate:
// - ErrorCode::kBinaryTooLarge enum value
// - absl::Status BinaryTooLarge(...) factory function.
// These can be used in the XLA codebase to uniquely identify errors and to
// generate absl::Status objects with the correct error code respectively.

#define XLA_ERROR_CODE_LIST(X)                                          \
  /* go/keep-sorted start */                                            \
  /* E00xx - Generic Error Codes mimicking absl::Status codes. */       \
  X("E0000", Cancelled, absl::StatusCode::kCancelled)                   \
  X("E0001", Unknown, absl::StatusCode::kUnknown)                       \
  X("E0002", InvalidArgument, absl::StatusCode::kInvalidArgument)       \
  X("E0003", DeadlineExceeded, absl::StatusCode::kDeadlineExceeded)     \
  X("E0004", NotFound, absl::StatusCode::kNotFound)                     \
  X("E0005", AlreadyExists, absl::StatusCode::kAlreadyExists)           \
  X("E0006", PermissionDenied, absl::StatusCode::kPermissionDenied)     \
  X("E0007", ResourceExhausted, absl::StatusCode::kResourceExhausted)   \
  X("E0008", FailedPrecondition, absl::StatusCode::kFailedPrecondition) \
  X("E0009", Aborted, absl::StatusCode::kAborted)                       \
  X("E0010", OutOfRange, absl::StatusCode::kOutOfRange)                 \
  X("E0011", Unimplemented, absl::StatusCode::kUnimplemented)           \
  X("E0012", Internal, absl::StatusCode::kInternal)                     \
  X("E0013", Unavailable, absl::StatusCode::kUnavailable)               \
  X("E0014", DataLoss, absl::StatusCode::kDataLoss)                     \
  X("E0015", Unauthenticated, absl::StatusCode::kUnauthenticated)       \
                                                                        \
  /* E1xxx - Compile Time OOM Error Codes */                            \
                                                                        \
  /* E2xxx - Compile Time Internal Error Codes */                       \
                                                                        \
  /* E3xxx - Compile Time Unimplemented Error Codes */                  \
                                                                        \
  /* E41xx - Compile Time Mosaic Internal Error Codes */                \
                                                                        \
  /* E42xx - Compile Time Mosaic User Error Codes */                    \
                                                                        \
  /* E43xx - Compile Time Mosaic Unimplemented Error Codes */           \
                                                                        \
  /* E4xxx - Compile Time Mosaic Deserialization Error Codes */         \
                                                                        \
  /* E50xx - Resource Exhausted, Memory Allocation Error Codes */       \
                                                                        \
  /* E51xx - Compiler Inserted Scheck(TC)/Assert(SC) Error Codes */     \
                                                                        \
  /* E52xx - Hardware Detected Program/User Errors Codes */             \
                                                                        \
  /* E53xx - Hardware/Network/Power Fatal Error Codes */                \
                                                                        \
  /* E54xx - Runtime Initiated Cancellations Error Codes */             \
                                                                        \
  /* E6xxx - Megascale, XLA “extensions” Error Codes */                 \
                                                                        \
  /* go/keep-sorted end */

// Enum that enumerates all XLA error codes.
enum class ErrorCode {
#define DECLARE_ERROR_CODE_ENUM(string_id, enum_name, ...) k##enum_name,
  XLA_ERROR_CODE_LIST(DECLARE_ERROR_CODE_ENUM)
#undef DECLARE_ERROR_CODE_ENUM
};

// === HELPER FUNCTIONS ===

// Returns the canonical identifier of the error code. This is the identifier
// that is used in the XLA documentation to link to the documentation page for
// the error code.
inline absl::string_view ErrorCodeToStringIdentifier(ErrorCode code) {
  switch (code) {
#define GET_STRING_IDENTIFIER_CASE(string_id, enum_name, ...) \
  case ErrorCode::k##enum_name:                               \
    return string_id;
    XLA_ERROR_CODE_LIST(GET_STRING_IDENTIFIER_CASE)
#undef GET_STRING_IDENTIFIER_CASE
  }
  return "UNKNOWN_CODE";
}

// Returns the human readable name of the error code.
inline absl::string_view ErrorCodeToName(ErrorCode code) {
  switch (code) {
#define GET_NAME_CASE(string_id, enum_name, _) \
  case ErrorCode::k##enum_name:                \
    return #enum_name;
    XLA_ERROR_CODE_LIST(GET_NAME_CASE)
#undef GET_NAME_CASE
  }
  return "An unknown error occurred.";
}

// Returns the error code and name as a string, e.g. "E0000: Cancelled".
inline std::string GetErrorCodeAndName(ErrorCode code) {
  return absl::StrCat(ErrorCodeToStringIdentifier(code), ": ",
                      ErrorCodeToName(code));
}

// Generates a URL for the error's documentation page.
// A string like "https://openxla.org/xla/errors/E0000".
inline std::string GetErrorUrl(ErrorCode code) {
  return absl::StrCat("https://openxla.org/xla/errors/",
                      ErrorCodeToStringIdentifier(code));
}

// === ErrorCode specific absl::Status factory functions ===
// Automatically generates a factory function for each error code
// to easily create a properly formatted absl::Status object.
// e.g. can be used as:
// error::InvalidArgument("Compilation failed with error: %s", my_message);
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code) \
  template <typename... Args>                                            \
  inline absl::Status enum_name(const absl::FormatSpec<Args...>& format, \
                                const Args&... args) {                   \
    auto status = absl::Status(                                          \
        status_code,                                                     \
        absl::StrCat(GetErrorCodeAndName(ErrorCode::k##enum_name), ": ", \
                     absl::StrFormat(format, args...)));                 \
    error::AttachDebugMeContextPayload(status);                          \
    return status;                                                       \
  }

XLA_ERROR_CODE_LIST(DEFINE_ERROR_FACTORY_FUNCTION)
#undef DEFINE_ERROR_FACTORY_FUNCTION

}  // namespace xla::error

#endif  // XLA_ERROR_ERROR_CODES_H_
