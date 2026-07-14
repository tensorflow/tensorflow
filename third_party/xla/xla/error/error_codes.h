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
//   XLA error messages.
// - absl::StatusCode is the canonical absl::StatusCode that most closely
//   matches the error code.
// e.g. For a declaration: X("E1234", BinaryTooLarge, absl::kResourceExhausted)
// the following is automatically generated:
// - A factory function BinaryTooLarge(...) that can be used as a drop-in
//   replacement for absl::ResourceExhaustedError(...). This function creates
//   and returns an absl::Status object with an error message prefixed with
//   "E1234: BinaryTooLarge" and suffixed with a URL to the XLA documentation
//   for this error code.
// - an Enum value ErrorCode::kBinaryTooLarge that can be used in the XLA
// codebase to uniquely identify error sources.

#define XLA_ERROR_CODE_LIST(X)                                                \
  /* go/keep-sorted start */                                                  \
  /* E00xx - Generic Error Codes mimicking absl::Status codes. */             \
  X("E0000", Cancelled, absl::StatusCode::kCancelled)                         \
  X("E0001", Unknown, absl::StatusCode::kUnknown)                             \
  X("E0002", InvalidArgument, absl::StatusCode::kInvalidArgument)             \
  X("E0003", DeadlineExceeded, absl::StatusCode::kDeadlineExceeded)           \
  X("E0004", NotFound, absl::StatusCode::kNotFound)                           \
  X("E0005", AlreadyExists, absl::StatusCode::kAlreadyExists)                 \
  X("E0006", PermissionDenied, absl::StatusCode::kPermissionDenied)           \
  X("E0007", ResourceExhausted, absl::StatusCode::kResourceExhausted)         \
  X("E0008", FailedPrecondition, absl::StatusCode::kFailedPrecondition)       \
  X("E0009", Aborted, absl::StatusCode::kAborted)                             \
  X("E0010", OutOfRange, absl::StatusCode::kOutOfRange)                       \
  X("E0011", Unimplemented, absl::StatusCode::kUnimplemented)                 \
  X("E0012", Internal, absl::StatusCode::kInternal)                           \
  X("E0013", Unavailable, absl::StatusCode::kUnavailable)                     \
  X("E0014", DataLoss, absl::StatusCode::kDataLoss)                           \
  X("E0015", Unauthenticated, absl::StatusCode::kUnauthenticated)             \
                                                                              \
  X("E0100", RuntimeBufferAllocationFailure,                                  \
    absl::StatusCode::kResourceExhausted)                                     \
  X("E0101", RuntimeProgramAllocationFailure,                                 \
    absl::StatusCode::kResourceExhausted)                                     \
  X("E0102", RuntimeProgramInputMismatch, absl::StatusCode::kInvalidArgument) \
  X("E0200", RuntimeUnexpectedCoreHalt, absl::StatusCode::kInternal)          \
  X("E1000", CompileTimeHbmOom, absl::StatusCode::kResourceExhausted)         \
  X("E1001", CompileTimeScopedVmemOom, absl::StatusCode::kResourceExhausted)  \
  X("E1200", CompileTimeHostOffloadOutputLocationMismatch,                    \
    absl::StatusCode::kInvalidArgument)                                       \
  X("E2001", CompileTimeMosaicUnsupportedRhsType,                             \
    absl::StatusCode::kUnimplemented)                                         \
  X("E2002", CompileTimeMosaicMisalignedBlockAndTiling,                       \
    absl::StatusCode::kInvalidArgument)                                       \
  X("E2003", CompileTimeMosaicUnprovenMemoryAccessAlignment,                  \
    absl::StatusCode::kInvalidArgument)                                       \
  X("E3000", CompileTimeSparseCoreAllocationFailure,                          \
    absl::StatusCode::kResourceExhausted)                                     \
  X("E3001", CompileTimeSparseCoreInvalidReplicaCount,                        \
    absl::StatusCode::kInvalidArgument)                                       \
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
// A string like "https://openxla.org/xla/errors/error_0000".
inline std::string GetErrorUrl(ErrorCode code) {
  absl::string_view id = ErrorCodeToStringIdentifier(code);
  if (!id.empty() && id.front() == 'E') {
    id.remove_prefix(1);
  }

  return absl::StrCat("https://openxla.org/xla/errors/error_", id);
}

// Returns the error message with the standard XLA Error Code formatting:
// "EXXXX: ErrorName:\n<Original Message>\nSee <URL> for more details."
inline std::string FormatMessageWithCode(absl::string_view message,
                                         ErrorCode code) {
  return absl::StrCat(GetErrorCodeAndName(code), ": ", message, "\nSee ",
                      GetErrorUrl(code), " for more details.");
}

// Wraps an existing status with the standard XLA Error Code formatting:
// "EXXXX: ErrorName:\n<Original Message>\nSee <URL> for more details."
inline absl::Status AnnotateWithCode(const absl::Status& original,
                                     ErrorCode code) {
  if (original.ok()) {
    return original;
  }

  absl::Status new_status(original.code(),
                          FormatMessageWithCode(original.message(), code));

  // Copy over the payloads and source locations from the original status.
  original.ForEachPayload(
      [&new_status](absl::string_view type_url, const absl::Cord& payload) {
        new_status.SetPayload(type_url, payload);
      });

#if defined(PLATFORM_GOOGLE)
  for (const auto& loc : original.GetSourceLocations()) {
    new_status.AddSourceLocation(loc);
  }
#endif  // PLATFORM_GOOGLE

  return new_status;
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

// Generates a factory function e.g. BinaryTooLarge(...) that can be used as a
// drop-in replacement for absl::ResourceExhaustedError(...).
// For example: error::BinaryTooLarge("Something was wrong: %s", "My message");
// returns an absl::Status object with the following error message:
// "E1234: BinaryTooLarge:
// Something was wrong: My message
// See https://openxla.org/xla/errors#e1234 for more details.".
#if defined(PLATFORM_GOOGLE)
// This version captures the caller's source location and attaches it to the
// absl::Status object on platforms that support it.
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code)       \
  DEFINE_ERROR_FACTORY_FUNCTION_PREFIX(enum_name)                              \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                            \
  enum_name(const absl::FormatSpec<Args...>& format, Args&&... args,           \
            absl::SourceLocation location = absl::SourceLocation::current())   \
      : status(                                                                \
            status_code,                                                       \
            absl::StrCat(GetErrorCodeAndName(ErrorCode::k##enum_name), ":\n",  \
                         absl::StrFormat(format, std::forward<Args>(args)...), \
                         "\nSee ", GetErrorUrl(ErrorCode::k##enum_name),       \
                         " for more details."),                                \
            location) {}                                                       \
  DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX(enum_name)
#else  // !PLATFORM_GOOGLE
// Absl::SourceLocation is not yet open-source. This version does NOT capture
// the source location.
#define DEFINE_ERROR_FACTORY_FUNCTION(string_id, enum_name, status_code)       \
  DEFINE_ERROR_FACTORY_FUNCTION_PREFIX(enum_name)                              \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                            \
  enum_name(const absl::FormatSpec<Args...>& format, Args&&... args)           \
      : status(                                                                \
            status_code,                                                       \
            absl::StrCat(GetErrorCodeAndName(ErrorCode::k##enum_name), ":\n",  \
                         absl::StrFormat(format, std::forward<Args>(args)...), \
                         "\nSee ", GetErrorUrl(ErrorCode::k##enum_name),       \
                         " for more details.")) {}                             \
  DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX(enum_name)
#endif  // PLATFORM_GOOGLE

XLA_ERROR_CODE_LIST(DEFINE_ERROR_FACTORY_FUNCTION)

#undef DEFINE_ERROR_FACTORY_FUNCTION_PREFIX
#undef DEFINE_ERROR_FACTORY_FUNCTION_SUFFIX
#undef DEFINE_ERROR_FACTORY_FUNCTION

}  // namespace xla::error

#endif  // XLA_ERROR_ERROR_CODES_H_
