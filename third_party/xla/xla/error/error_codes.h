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
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/error/debug_me_context_util.h"
#include "tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
#include "absl/types/source_location.h"
#endif  // PLATFORM_GOOGLE

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
// A string like "https://openxla.org/xla/errors#e0000".
inline std::string GetErrorUrl(ErrorCode code) {
  return absl::StrCat("https://openxla.org/xla/errors#",
                      absl::AsciiStrToLower(ErrorCodeToStringIdentifier(code)));
}

// The following three macros implement a factory pattern for creating
// absl::Status objects. This pattern is necessary to reliably capture the
// caller's source location while supporting variadic arguments.
//
// It is split into three parts (PREFIX, SUFFIX, and the main macro) for
// modularity and readability - separating the boilerplate of the
// struct definition from the implementation of its constructor.
#define DEFINE_ERROR_FACTORY_FUNCTION_PREFIX(enum_name) \
  template <typename... Args>                           \
  struct enum_name {                                    \
    absl::Status status;

// The SUFFIX macro defines the end of the error-generating struct. It provides
// A class template deduction guide (CTAD), which is essential for making the
// factory's calling syntax work. It tells the compiler how to deduce the
// template arguments `Args...` from the constructor call.
#define DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX(enum_name)                       \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                           \
  operator absl::Status() const& { return status; }                           \
  operator absl::Status() && { return std::move(status); }                    \
  }                                                                           \
  ;                                                                           \
  /* Deduction guide to make variadic arguments play nice with the default */ \
  /* absl::SourceLocation argument. */                                        \
  template <typename... Args>                                                 \
  enum_name(const absl::FormatSpec<Args...>&, Args&&...)                      \
      -> enum_name<Args...>;

// Main macro that generates a status factory function for each error code that
// can be used to create an absl::Status object with an error code and a
// formatted error message. Additionally attaches a payload with DebugMeContext
// details if present. e.g. can be used as: error::InvalidArgument("Compilation
// failed with error: %s", my_message);
#if defined(PLATFORM_GOOGLE)
// This version captures the caller's source location and attaches it to the
// absl::Status object on platforms that support it.
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code)      \
  DEFINE_ERROR_FACTORY_FUNCTION_PREFIX(enum_name)                             \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                           \
  enum_name(const absl::FormatSpec<Args...>& format, Args&&... args,          \
            absl::SourceLocation location = absl::SourceLocation::current())  \
      : status([&] {                                                          \
          auto s = absl::Status(                                              \
              status_code,                                                    \
              absl::StrCat(                                                   \
                  GetErrorCodeAndName(ErrorCode::k##enum_name), ":\n",        \
                  absl::StrFormat(format, std::forward<Args>(args)...), "\n", \
                  GetErrorUrl(ErrorCode::k##enum_name)),                      \
              location);                                                      \
          error::AttachDebugMeContextPayload(s);                              \
          return s;                                                           \
        }()) {}                                                               \
  DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX(enum_name)
#else  // !PLATFORM_GOOGLE
// Absl::SourceLocation is not yet open-source. This version does NOT capture
// the source location.
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code)      \
  DEFINE_ERROR_FACTORY_FUNCTION_PREFIX(enum_name)                             \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                           \
  enum_name(const absl::FormatSpec<Args...>& format, Args&&... args)          \
      : status([&] {                                                          \
          auto s = absl::Status(                                              \
              status_code,                                                    \
              absl::StrCat(                                                   \
                  GetErrorCodeAndName(ErrorCode::k##enum_name), ":\n",        \
                  absl::StrFormat(format, std::forward<Args>(args)...), "\n", \
                  GetErrorUrl(ErrorCode::k##enum_name)));                     \
          error::AttachDebugMeContextPayload(s);                              \
          return s;                                                           \
        }()) {}                                                               \
  DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX(enum_name)
#endif  // PLATFORM_GOOGLE

XLA_ERROR_CODE_LIST(DEFINE_ERROR_FACTORY_FUNCTION)

#undef DEFINE_ERROR_FACTORY_FUNCTION_PREFIX
#undef DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX
#undef DEFINE_ERROR_FACTORY_FUNCTION

}  // namespace xla::error

#endif  // XLA_ERROR_ERROR_CODES_H_
