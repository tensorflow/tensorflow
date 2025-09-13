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

#ifndef XLA_ERRORS_ERROR_CODES_H_
#define XLA_ERRORS_ERROR_CODES_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace xla {

// The single source of truth for all XLA error codes. These error codes provide
// a consistent way to report errors across XLA. They provide finer grained
// error codes than absl::Status codes and each error code is linked to a
// corresponding page in the XLA documentation with more details about the error
// and potential solutions.

// The format of the macro declaration is:
// X(ErrorCode, ErrorCodeEnumName, absl::StatusCode, ErrorName)
// where:
// - ErrorCode is the canonical identifier of the error code - all XLA
//   documentation is indexed by this identifier.
// - ErrorCodeEnumName is the enum name for the error code used internally in
//   XLA.
// - absl::StatusCode is the canonical absl::StatusCode that most closely
//   matches the error code, and
// - ErrorName is a human readable name for the error code.
#define XLA_ERROR_CODE_LIST(X)                                                 \
  /* --- Compilation Failures --- */                                           \
  X("C0000", CompileCancelled, absl::StatusCode::kCancelled,                   \
    "Compilation Failure: Cancelled")                                          \
  X("C0001", CompileUnknown, absl::StatusCode::kUnknown,                       \
    "Compilation Failure: Unknown")                                            \
  X("C0002", CompileInvalidArgument, absl::StatusCode::kInvalidArgument,       \
    "Compilation Failure: Invalid Argument")                                   \
  X("C0003", CompileDeadlineExceeded, absl::StatusCode::kDeadlineExceeded,     \
    "Compilation Failure: Deadline Exceeded")                                  \
  X("C0004", CompileNotFound, absl::StatusCode::kNotFound,                     \
    "Compilation Failure: Not Found")                                          \
  X("C0005", CompileAlreadyExists, absl::StatusCode::kAlreadyExists,           \
    "Compilation Failure: Already Exists")                                     \
  X("C0006", CompilePermissionDenied, absl::StatusCode::kPermissionDenied,     \
    "Compilation Failure: Permission Denied")                                  \
  X("C0007", CompileResourceExhausted, absl::StatusCode::kResourceExhausted,   \
    "Compilation Failure: Resource Exhausted")                                 \
  X("C0008", CompileFailedPrecondition, absl::StatusCode::kFailedPrecondition, \
    "Compilation Failure: Failed Precondition")                                \
  X("C0009", CompileAborted, absl::StatusCode::kAborted,                       \
    "Compilation Failure: Aborted")                                            \
  X("C0010", CompileOutOfRange, absl::StatusCode::kOutOfRange,                 \
    "Compilation Failure: Out of Range")                                       \
  X("C0011", CompileUnimplemented, absl::StatusCode::kUnimplemented,           \
    "Compilation Failure: Unimplemented")                                      \
  X("C0012", CompileInternal, absl::StatusCode::kInternal,                     \
    "Compilation Failure: Internal Error")                                     \
  X("C0013", CompileUnavailable, absl::StatusCode::kUnavailable,               \
    "Compilation Failure: Unavailable")                                        \
  X("C0014", CompileDataLoss, absl::StatusCode::kDataLoss,                     \
    "Compilation Failure: Data Loss")                                          \
  X("C0015", CompileUnauthenticated, absl::StatusCode::kUnauthenticated,       \
    "Compilation Failure: Unauthenticated")                                    \
                                                                               \
  /* C1xxx - Compile Time OOM */                                               \
                                                                               \
  /* C2xxx - Compile Time Internal Error */                                    \
                                                                               \
  /* C3xxx - Compile Time Unimplemented Error */                               \
                                                                               \
  /* C4xxx - Compile Time Mosaic Deserialization Error */                      \
                                                                               \
  /* C41xx - Compile Time Mosaic Internal Error */                             \
                                                                               \
  /* C42xx - Compile Time Mosaic User Error */                                 \
                                                                               \
  /* C43xx - Compile Time Mosaic Unimplemented Error */                        \
                                                                               \
  /* --- Runtime Failures --- */                                               \
  X("R0000", RuntimeCancelled, absl::StatusCode::kCancelled,                   \
    "Runtime Failure: Cancelled")                                              \
  X("R0001", RuntimeUnknown, absl::StatusCode::kUnknown,                       \
    "Runtime Failure: Unknown")                                                \
  X("R0002", RuntimeInvalidArgument, absl::StatusCode::kInvalidArgument,       \
    "Runtime Failure: Invalid Argument")                                       \
  X("R0003", RuntimeDeadlineExceeded, absl::StatusCode::kDeadlineExceeded,     \
    "Runtime Failure: Deadline Exceeded")                                      \
  X("R0004", RuntimeNotFound, absl::StatusCode::kNotFound,                     \
    "Runtime Failure: Not Found")                                              \
  X("R0005", RuntimeAlreadyExists, absl::StatusCode::kAlreadyExists,           \
    "Runtime Failure: Already Exists")                                         \
  X("R0006", RuntimePermissionDenied, absl::StatusCode::kPermissionDenied,     \
    "Runtime Failure: Permission Denied")                                      \
  X("R0007", RuntimeResourceExhausted, absl::StatusCode::kResourceExhausted,   \
    "Runtime Failure: Resource Exhausted")                                     \
  X("R0008", RuntimeFailedPrecondition, absl::StatusCode::kFailedPrecondition, \
    "Runtime Failure: Failed Precondition")                                    \
  X("R0009", RuntimeAborted, absl::StatusCode::kAborted,                       \
    "Runtime Failure: Aborted")                                                \
  X("R0010", RuntimeOutOfRange, absl::StatusCode::kOutOfRange,                 \
    "Runtime Failure: Out of Range")                                           \
  X("R0011", RuntimeUnimplemented, absl::StatusCode::kUnimplemented,           \
    "Runtime Failure: Unimplemented")                                          \
  X("R0012", RuntimeInternal, absl::StatusCode::kInternal,                     \
    "Runtime Failure: Internal Error")                                         \
  X("R0013", RuntimeUnavailable, absl::StatusCode::kUnavailable,               \
    "Runtime Failure: Unavailable")                                            \
  X("R0014", RuntimeDataLoss, absl::StatusCode::kDataLoss,                     \
    "Runtime Failure: Data Loss")                                              \
  X("R0015", RuntimeUnauthenticated, absl::StatusCode::kUnauthenticated,       \
    "Runtime Failure: Unauthenticated")                                        \
                                                                               \
  /* R10xx - Resource Exhausted, Memory Allocation Failures */                 \
                                                                               \
  /* R2xxx - Compiler Inserted Scheck(TC)/Assert(SC) */                        \
                                                                               \
  /* R3xxx - Hardware Detected Program/User Errors */                          \
                                                                               \
  /* R4xxx - Hardware/Network/Power Fatal Error */                             \
                                                                               \
  /* R41xx - Runtime Initiated Cancellations */                                \
                                                                               \
  /* R42xx - Megascale, XLA “extensions” Failures */

// Enum that enumerates all XLA error codes.
enum class ErrorCode {
#define DECLARE_ERROR_CODE_ENUM(string_id, enum_name, ...) enum_name,
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
  case ErrorCode::enum_name:                                  \
    return string_id;
    XLA_ERROR_CODE_LIST(GET_STRING_IDENTIFIER_CASE)
#undef GET_STRING_IDENTIFIER_CASE
  }
  return "UNKNOWN_CODE";
}

// Returns the human readable name of the error code.
inline absl::string_view ErrorCodeToName(ErrorCode code) {
  switch (code) {
#define GET_DESCRIPTION_CASE(string_id, enum_name, _, description) \
  case ErrorCode::enum_name:                                       \
    return description;
    XLA_ERROR_CODE_LIST(GET_DESCRIPTION_CASE)
#undef GET_DESCRIPTION_CASE
  }
  return "An unknown error occurred.";
}

// Returns the error code and name as a string, e.g. "C0000: Compilation
// Failure: Cancelled".
inline std::string GetErrorCodeAndName(ErrorCode code) {
  return absl::StrCat(ErrorCodeToStringIdentifier(code), ": ",
                      ErrorCodeToName(code));
}

/**
 * @brief Generates a URL for the error's documentation page.
 * @return A string like "https://openxla.org/xla/errors/C0000".
 */
inline std::string GetErrorUrl(ErrorCode code) {
  return absl::StrCat("https://openxla.org/xla/errors/",
                      ErrorCodeToStringIdentifier(code));
}

// === ErrorCode specific absl::Status factory functions ===
// Automatically generates a factory function for each error code
// to easily create a properly formatted absl::Status object.
// e.g. can be used as:
// CompileInvalidArgument("Compilation failed with error: %s",
//                               my_error_message);
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code,     \
                                      description)                           \
  template <typename... Args>                                                \
  inline absl::Status enum_name(const absl::FormatSpec<Args...>& format,     \
                                const Args&... args) {                       \
    return absl::Status(                                                     \
        status_code, absl::StrCat(GetErrorCodeAndName(ErrorCode::enum_name), \
                                  ": ", absl::StrFormat(format, args...)));  \
  }

XLA_ERROR_CODE_LIST(DEFINE_ERROR_FACTORY_FUNCTION)
#undef DEFINE_ERROR_FACTORY_FUNCTION

}  // namespace xla

#endif  // XLA_ERRORS_ERROR_CODES_H_
