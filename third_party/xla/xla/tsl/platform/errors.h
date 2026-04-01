/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_ERRORS_H_
#define XLA_TSL_PLATFORM_ERRORS_H_

#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/strcat.h"

namespace tsl {
namespace error {
// NOLINTBEGIN(misc-unused-using-decls)
// TODO(aminim): figure out the protobuf migration story.
using tensorflow::error::ABORTED;
using tensorflow::error::ALREADY_EXISTS;
using tensorflow::error::CANCELLED;
using tensorflow::error::Code;
using tensorflow::error::DATA_LOSS;
using tensorflow::error::DEADLINE_EXCEEDED;
using tensorflow::error::FAILED_PRECONDITION;
using tensorflow::error::INTERNAL;
using tensorflow::error::INVALID_ARGUMENT;
using tensorflow::error::NOT_FOUND;
using tensorflow::error::OK;
using tensorflow::error::OUT_OF_RANGE;
using tensorflow::error::PERMISSION_DENIED;
using tensorflow::error::RESOURCE_EXHAUSTED;
using tensorflow::error::UNAUTHENTICATED;
using tensorflow::error::UNAVAILABLE;
using tensorflow::error::UNIMPLEMENTED;
using tensorflow::error::UNKNOWN;
// NOLINTEND(misc-unused-using-decls)
}  // namespace error

namespace errors {

namespace internal {

// The DECLARE_ERROR macro below only supports types that can be converted
// into StrCat's AlphaNum. For the other types we rely on a slower path
// through std::stringstream. To add support of a new type, it is enough to
// make sure there is an operator<<() for it:
//
//   std::ostream& operator<<(std::ostream& os, const MyType& foo) {
//     os << foo.ToString();
//     return os;
//   }
// Eventually absl::strings will have native support for this and we will be
// able to completely remove PrepareForStrCat().
template <typename T>
typename std::enable_if<!std::is_convertible<T, absl::AlphaNum>::value,
                        std::string>::type
PrepareForStrCat(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}
inline const absl::AlphaNum& PrepareForStrCat(const absl::AlphaNum& a) {
  return a;
}
// Helper trait to check if all types in a pack are convertible to
// absl::AlphaNum
template <typename... Args>
struct all_alphanum_convertible {
  static constexpr bool value =
      (std::is_convertible_v<std::decay_t<Args>, absl::AlphaNum> && ...);
};
}  // namespace internal

#define ABSL_STATUS                                            \
  typename std::enable_if<                                     \
      sizeof...(Args) == 12 || sizeof...(Args) == 13 ||        \
          sizeof...(Args) == 14 || sizeof...(Args) == 15 ||    \
          sizeof...(Args) == 16 || sizeof...(Args) == 17 ||    \
          sizeof...(Args) == 18 ||                             \
          !internal::all_alphanum_convertible<Args...>::value, \
      absl::Status>::type

// Maps UNIX errors into a Status.
absl::Status IOError(absl::string_view context, int err_number);

// Returns all payloads from a Status as a key-value map.
inline std::unordered_map<std::string, std::string> GetPayloads(
    const absl::Status& status) {
  std::unordered_map<std::string, std::string> payloads;
  status.ForEachPayload(
      [&payloads](absl::string_view key, const absl::Cord& value) {
        payloads[std::string(key)] = std::string(value);
      });
  return payloads;
}

// Inserts all given payloads into the given status. Will overwrite existing
// payloads if they exist with the same key.
inline void InsertPayloads(
    absl::Status& status,
    const std::unordered_map<std::string, std::string>& payloads) {
  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, absl::Cord(payload.second));
  }
}

// Copies all payloads from one Status to another. Will overwrite existing
// payloads in the destination if they exist with the same key.
inline void CopyPayloads(const absl::Status& from, absl::Status& to) {
  from.ForEachPayload([&to](absl::string_view key, const absl::Cord& value) {
    to.SetPayload(key, value);
  });
}

#if defined(PLATFORM_GOOGLE)
// Creates a new status with the given code, message and payloads.
inline absl::Status Create(
    absl::StatusCode code, absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads,
    absl::SourceLocation loc = absl::SourceLocation::current()) {
  absl::Status status(code, message, loc);
  InsertPayloads(status, payloads);
  return status;
}
// Returns a new Status, replacing its message with the given.
inline absl::Status CreateWithUpdatedMessage(const absl::Status& status,
                                             absl::string_view message) {
  auto locations = status.GetSourceLocations();
  auto initial_loc =
      locations.empty() ? absl::SourceLocation::current() : locations[0];
  absl::Status new_status = Create(static_cast<absl::StatusCode>(status.code()),
                                   message, GetPayloads(status), initial_loc);
  if (locations.size() > 1) {
    for (auto loc : locations.subspan(1)) {
      new_status.AddSourceLocation(loc);
    }
  }
  return new_status;
}

#else
inline absl::Status Create(
    absl::StatusCode code, absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  Status status(code, message);
  InsertPayloads(status, payloads);
  return status;
}
// Returns a new Status, replacing its message with the given.
inline absl::Status CreateWithUpdatedMessage(const absl::Status& status,
                                             absl::string_view message) {
  return Create(static_cast<absl::StatusCode>(status.code()), message,
                GetPayloads(status));
}
#endif

// Append some context to an error message.  Each time we append
// context put it on a new line, since it is possible for there
// to be several layers of additional context.
template <typename... Args>
void AppendToMessage(absl::Status* status, Args... args) {
  auto new_status = CreateWithUpdatedMessage(
      *status, ::tsl::strings::StrCat(status->message(), "\n\t", args...));
  CopyPayloads(*status, new_status);
  *status = std::move(new_status);
}

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)            \
  do {                                     \
    absl::Status _status = (__VA_ARGS__);  \
    if (TF_PREDICT_FALSE(!_status.ok())) { \
      MAYBE_ADD_SOURCE_LOCATION(_status)   \
      return _status;                      \
    }                                      \
  } while (0)

#define TF_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)           \
  do {                                                       \
    absl::Status _status = (expr);                           \
    if (TF_PREDICT_FALSE(!_status.ok())) {                   \
      ::tsl::errors::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                        \
    }                                                        \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#if !defined(ABSL_REFACTOR_INLINE)
#define ABSL_REFACTOR_INLINE
#define ABSL_REFACTOR_INLINE_DEFINED_LOCALLY
#endif

// CANCELLED
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::Status& arg) {
  return absl::CancelledError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::CancelledError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Cancelled(Arg1 arg1) {
  return absl::CancelledError(arg1);
}

ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2) {
  return absl::CancelledError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2,
                              const absl::AlphaNum& arg3) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2,
                              const absl::AlphaNum& arg3,
                              const absl::AlphaNum& arg4) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2,
                              const absl::AlphaNum& arg3,
                              const absl::AlphaNum& arg4,
                              const absl::AlphaNum& arg5) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2,
                              const absl::AlphaNum& arg3,
                              const absl::AlphaNum& arg4,
                              const absl::AlphaNum& arg5,
                              const absl::AlphaNum& arg6) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(const absl::AlphaNum& arg1,
                              const absl::AlphaNum& arg2,
                              const absl::AlphaNum& arg3,
                              const absl::AlphaNum& arg4,
                              const absl::AlphaNum& arg5,
                              const absl::AlphaNum& arg6,
                              const absl::AlphaNum& arg7) {
  return absl::CancelledError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::CancelledError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATED("Use absl::CancelledError(absl::StrCat(args...)) instead.")
inline absl::Status Cancelled(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::CancelledError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                           arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Cancelled(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::CancelledError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                           arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::CancelledError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Cancelled(Args... rest) {
  return absl::CancelledError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
template <typename... Args>
absl::Status CancelledWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kCancelled, message, payloads);
}

// InvalidArgument
// Function for the cases where we need to have a small call stack footprint.
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::InvalidArgumentError(absl::StrCat(args...)) instead.")
absl::Status InvalidArgumentError(Args... args) {
  return absl::InvalidArgumentError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(args)...));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::Status& arg) { return arg; }
template <typename Arg1>
ABSL_DEPRECATED("Use absl::InvalidArgumentError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status InvalidArgument(Arg1 arg1) {
  return absl::InvalidArgumentError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2) {
  return absl::InvalidArgumentError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3) {
  return absl::InvalidArgumentError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4) {
  return absl::InvalidArgumentError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4,
                                    const absl::AlphaNum& arg5) {
  return absl::InvalidArgumentError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4,
                                    const absl::AlphaNum& arg5,
                                    const absl::AlphaNum& arg6) {
  return absl::InvalidArgumentError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4,
                                    const absl::AlphaNum& arg5,
                                    const absl::AlphaNum& arg6,
                                    const absl::AlphaNum& arg7) {
  return absl::InvalidArgumentError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::InvalidArgumentError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::InvalidArgumentError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::InvalidArgumentError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status InvalidArgument(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::InvalidArgumentError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::InvalidArgumentError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS InvalidArgument(Args... rest) {
  return absl::InvalidArgumentError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status InvalidArgumentWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kInvalidArgument, message, payloads);
}

// NotFound
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::Status& arg) {
  return absl::NotFoundError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::NotFoundError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status NotFound(Arg1 arg1) {
  return absl::NotFoundError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6,
                             const absl::AlphaNum& arg7) {
  return absl::NotFoundError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::NotFoundError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::NotFoundError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status NotFound(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::NotFoundError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::NotFoundError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS NotFound(Args... rest) {
  return absl::NotFoundError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status NotFoundWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kNotFound, message, payloads);
}

// AlreadyExists
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::Status& arg) {
  return absl::AlreadyExistsError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::AlreadyExistsError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status AlreadyExists(Arg1 arg1) {
  return absl::AlreadyExistsError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2) {
  return absl::AlreadyExistsError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3) {
  return absl::AlreadyExistsError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4) {
  return absl::AlreadyExistsError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5) {
  return absl::AlreadyExistsError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5,
                                  const absl::AlphaNum& arg6) {
  return absl::AlreadyExistsError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5,
                                  const absl::AlphaNum& arg6,
                                  const absl::AlphaNum& arg7) {
  return absl::AlreadyExistsError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::AlreadyExistsError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::AlreadyExistsError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::AlreadyExistsError(absl::StrCat(arg1, arg2, arg3, arg4, arg5,
                                               arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status AlreadyExists(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::AlreadyExistsError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::AlreadyExistsError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS AlreadyExists(Args... rest) {
  return absl::AlreadyExistsError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}

inline absl::Status AlreadyExistsWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kAlreadyExists, message, payloads);
}

// ResourceExhausted
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::Status& arg) {
  return absl::ResourceExhaustedError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::ResourceExhaustedError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status ResourceExhausted(Arg1 arg1) {
  return absl::ResourceExhaustedError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2) {
  return absl::ResourceExhaustedError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2,
                                      const absl::AlphaNum& arg3) {
  return absl::ResourceExhaustedError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2,
                                      const absl::AlphaNum& arg3,
                                      const absl::AlphaNum& arg4) {
  return absl::ResourceExhaustedError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2,
                                      const absl::AlphaNum& arg3,
                                      const absl::AlphaNum& arg4,
                                      const absl::AlphaNum& arg5) {
  return absl::ResourceExhaustedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2,
                                      const absl::AlphaNum& arg3,
                                      const absl::AlphaNum& arg4,
                                      const absl::AlphaNum& arg5,
                                      const absl::AlphaNum& arg6) {
  return absl::ResourceExhaustedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(const absl::AlphaNum& arg1,
                                      const absl::AlphaNum& arg2,
                                      const absl::AlphaNum& arg3,
                                      const absl::AlphaNum& arg4,
                                      const absl::AlphaNum& arg5,
                                      const absl::AlphaNum& arg6,
                                      const absl::AlphaNum& arg7) {
  return absl::ResourceExhaustedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::ResourceExhaustedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::ResourceExhaustedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::ResourceExhaustedError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status ResourceExhausted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::ResourceExhaustedError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::ResourceExhaustedError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS ResourceExhausted(Args... rest) {
  return absl::ResourceExhaustedError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status ResourceExhaustedWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kResourceExhausted, message,
                        payloads);
}

// Unavailable
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::Status& arg) {
  return absl::UnavailableError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::UnavailableError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Unavailable(Arg1 arg1) {
  return absl::UnavailableError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2,
                                const absl::AlphaNum& arg3) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2,
                                const absl::AlphaNum& arg3,
                                const absl::AlphaNum& arg4) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2,
                                const absl::AlphaNum& arg3,
                                const absl::AlphaNum& arg4,
                                const absl::AlphaNum& arg5) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2,
                                const absl::AlphaNum& arg3,
                                const absl::AlphaNum& arg4,
                                const absl::AlphaNum& arg5,
                                const absl::AlphaNum& arg6) {
  return absl::UnavailableError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(const absl::AlphaNum& arg1,
                                const absl::AlphaNum& arg2,
                                const absl::AlphaNum& arg3,
                                const absl::AlphaNum& arg4,
                                const absl::AlphaNum& arg5,
                                const absl::AlphaNum& arg6,
                                const absl::AlphaNum& arg7) {
  return absl::UnavailableError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::UnavailableError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::UnavailableError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                             arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unavailable(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::UnavailableError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                             arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::UnavailableError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Unavailable(Args... rest) {
  return absl::UnavailableError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status UnavailableWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kUnavailable, message, payloads);
}

// FailedPrecondition
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::Status& arg) {
  return absl::FailedPreconditionError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::FailedPreconditionError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status FailedPrecondition(Arg1 arg1) {
  return absl::FailedPreconditionError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2) {
  return absl::FailedPreconditionError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2,
                                       const absl::AlphaNum& arg3) {
  return absl::FailedPreconditionError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2,
                                       const absl::AlphaNum& arg3,
                                       const absl::AlphaNum& arg4) {
  return absl::FailedPreconditionError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2,
                                       const absl::AlphaNum& arg3,
                                       const absl::AlphaNum& arg4,
                                       const absl::AlphaNum& arg5) {
  return absl::FailedPreconditionError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2,
                                       const absl::AlphaNum& arg3,
                                       const absl::AlphaNum& arg4,
                                       const absl::AlphaNum& arg5,
                                       const absl::AlphaNum& arg6) {
  return absl::FailedPreconditionError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(const absl::AlphaNum& arg1,
                                       const absl::AlphaNum& arg2,
                                       const absl::AlphaNum& arg3,
                                       const absl::AlphaNum& arg4,
                                       const absl::AlphaNum& arg5,
                                       const absl::AlphaNum& arg6,
                                       const absl::AlphaNum& arg7) {
  return absl::FailedPreconditionError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::FailedPreconditionError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::FailedPreconditionError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::FailedPreconditionError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status FailedPrecondition(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::FailedPreconditionError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::FailedPreconditionError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS FailedPrecondition(Args... rest) {
  return absl::FailedPreconditionError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status FailedPreconditionWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kFailedPrecondition, message,
                        payloads);
}

// OutOfRange
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::Status& arg) {
  return absl::OutOfRangeError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::OutOfRangeError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status OutOfRange(Arg1 arg1) {
  return absl::OutOfRangeError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2,
                               const absl::AlphaNum& arg3) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2,
                               const absl::AlphaNum& arg3,
                               const absl::AlphaNum& arg4) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2,
                               const absl::AlphaNum& arg3,
                               const absl::AlphaNum& arg4,
                               const absl::AlphaNum& arg5) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2,
                               const absl::AlphaNum& arg3,
                               const absl::AlphaNum& arg4,
                               const absl::AlphaNum& arg5,
                               const absl::AlphaNum& arg6) {
  return absl::OutOfRangeError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(const absl::AlphaNum& arg1,
                               const absl::AlphaNum& arg2,
                               const absl::AlphaNum& arg3,
                               const absl::AlphaNum& arg4,
                               const absl::AlphaNum& arg5,
                               const absl::AlphaNum& arg6,
                               const absl::AlphaNum& arg7) {
  return absl::OutOfRangeError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::OutOfRangeError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::OutOfRangeError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                            arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status OutOfRange(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::OutOfRangeError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                            arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::OutOfRangeError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS OutOfRange(Args... rest) {
  return absl::OutOfRangeError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status OutOfRangeWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kOutOfRange, message, payloads);
}

// Unimplemented
template <typename... Args>
ABSL_DEPRECATED("Use absl::UnimplementedError(absl::StrCat(args...)) instead.")
absl::Status UnimplementedError(Args... args) {
  return absl::UnimplementedError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(args)...));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::Status& arg) {
  return absl::UnimplementedError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::UnimplementedError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Unimplemented(Arg1 arg1) {
  return absl::UnimplementedError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2) {
  return absl::UnimplementedError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3) {
  return absl::UnimplementedError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4) {
  return absl::UnimplementedError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5) {
  return absl::UnimplementedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5,
                                  const absl::AlphaNum& arg6) {
  return absl::UnimplementedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(const absl::AlphaNum& arg1,
                                  const absl::AlphaNum& arg2,
                                  const absl::AlphaNum& arg3,
                                  const absl::AlphaNum& arg4,
                                  const absl::AlphaNum& arg5,
                                  const absl::AlphaNum& arg6,
                                  const absl::AlphaNum& arg7) {
  return absl::UnimplementedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::UnimplementedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::UnimplementedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::UnimplementedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5,
                                               arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unimplemented(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::UnimplementedError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::UnimplementedError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Unimplemented(Args... rest) {
  return absl::UnimplementedError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status UnimplementedWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kUnimplemented, message, payloads);
}

// Internal
template <typename... Args>
ABSL_DEPRECATED("Use absl::InternalError(absl::StrCat(args...)) instead.")
absl::Status InternalError(Args... args) {
  return absl::InternalError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(args)...));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::Status& arg) {
  return absl::InternalError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::InternalError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Internal(Arg1 arg1) {
  return absl::InternalError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2) {
  return absl::InternalError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6,
                             const absl::AlphaNum& arg7) {
  return absl::InternalError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::InternalError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::InternalError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Internal(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::InternalError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::InternalError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Internal(Args... rest) {
  return absl::InternalError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status InternalWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kInternal, message, payloads);
}

// Aborted
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::Status& arg) {
  return absl::AbortedError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::AbortedError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Aborted(Arg1 arg1) {
  return absl::AbortedError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2) {
  return absl::AbortedError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5,
                            const absl::AlphaNum& arg6) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5,
                            const absl::AlphaNum& arg6,
                            const absl::AlphaNum& arg7) {
  return absl::AbortedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::AbortedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::AbortedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                         arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Aborted(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::AbortedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                         arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::AbortedError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Aborted(Args... rest) {
  return absl::AbortedError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status AbortedWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kAborted, message, payloads);
}

// DeadlineExceeded
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::Status& arg) {
  return absl::DeadlineExceededError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::DeadlineExceededError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status DeadlineExceeded(Arg1 arg1) {
  return absl::DeadlineExceededError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2) {
  return absl::DeadlineExceededError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3) {
  return absl::DeadlineExceededError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4) {
  return absl::DeadlineExceededError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5) {
  return absl::DeadlineExceededError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5,
                                     const absl::AlphaNum& arg6) {
  return absl::DeadlineExceededError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5,
                                     const absl::AlphaNum& arg6,
                                     const absl::AlphaNum& arg7) {
  return absl::DeadlineExceededError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::DeadlineExceededError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::DeadlineExceededError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::DeadlineExceededError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DeadlineExceeded(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::DeadlineExceededError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::DeadlineExceededError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS DeadlineExceeded(Args... rest) {
  return absl::DeadlineExceededError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status DeadlineExceededWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kDeadlineExceeded, message, payloads);
}

// DataLoss
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::Status& arg) {
  return absl::DataLossError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::DataLossError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status DataLoss(Arg1 arg1) {
  return absl::DataLossError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2) {
  return absl::DataLossError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(const absl::AlphaNum& arg1,
                             const absl::AlphaNum& arg2,
                             const absl::AlphaNum& arg3,
                             const absl::AlphaNum& arg4,
                             const absl::AlphaNum& arg5,
                             const absl::AlphaNum& arg6,
                             const absl::AlphaNum& arg7) {
  return absl::DataLossError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::DataLossError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::DataLossError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status DataLoss(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::DataLossError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                          arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::DataLossError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS DataLoss(Args... rest) {
  return absl::DataLossError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status DataLossWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kDataLoss, message, payloads);
}

// Unknown
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::Status& arg) {
  return absl::UnknownError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::UnknownError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Unknown(Arg1 arg1) {
  return absl::UnknownError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2) {
  return absl::UnknownError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5,
                            const absl::AlphaNum& arg6) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(const absl::AlphaNum& arg1,
                            const absl::AlphaNum& arg2,
                            const absl::AlphaNum& arg3,
                            const absl::AlphaNum& arg4,
                            const absl::AlphaNum& arg5,
                            const absl::AlphaNum& arg6,
                            const absl::AlphaNum& arg7) {
  return absl::UnknownError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::UnknownError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::UnknownError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                         arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unknown(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::UnknownError(absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6,
                                         arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED("Use absl::UnknownError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Unknown(Args... rest) {
  return absl::UnknownError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status UnknownPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kUnknown, message, payloads);
}
// PermissionDenied
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::Status& arg) {
  return absl::PermissionDeniedError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::PermissionDeniedError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status PermissionDenied(Arg1 arg1) {
  return absl::PermissionDeniedError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2) {
  return absl::PermissionDeniedError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3) {
  return absl::PermissionDeniedError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4) {
  return absl::PermissionDeniedError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5) {
  return absl::PermissionDeniedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5,
                                     const absl::AlphaNum& arg6) {
  return absl::PermissionDeniedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(const absl::AlphaNum& arg1,
                                     const absl::AlphaNum& arg2,
                                     const absl::AlphaNum& arg3,
                                     const absl::AlphaNum& arg4,
                                     const absl::AlphaNum& arg5,
                                     const absl::AlphaNum& arg6,
                                     const absl::AlphaNum& arg7) {
  return absl::PermissionDeniedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8) {
  return absl::PermissionDeniedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9) {
  return absl::PermissionDeniedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10) {
  return absl::PermissionDeniedError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status PermissionDenied(
    const absl::AlphaNum& arg1, const absl::AlphaNum& arg2,
    const absl::AlphaNum& arg3, const absl::AlphaNum& arg4,
    const absl::AlphaNum& arg5, const absl::AlphaNum& arg6,
    const absl::AlphaNum& arg7, const absl::AlphaNum& arg8,
    const absl::AlphaNum& arg9, const absl::AlphaNum& arg10,
    const absl::AlphaNum& arg11) {
  return absl::PermissionDeniedError(absl::StrCat(
      arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::PermissionDeniedError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS PermissionDenied(Args... rest) {
  return absl::PermissionDeniedError(
      absl::StrCat(::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status PermissionDeniedWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kPermissionDenied, message, payloads);
}

// Unauthenticated
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::Status& arg) {
  return absl::UnauthenticatedError(arg.message());
}
template <typename Arg1>
ABSL_DEPRECATED("Use absl::UnauthenticatedError(arg1) instead.")
ABSL_REFACTOR_INLINE inline absl::Status Unauthenticated(Arg1 arg1) {
  return absl::UnauthenticatedError(arg1);
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2) {
  return absl::UnauthenticatedError(absl::StrCat(arg1, arg2));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3) {
  return absl::UnauthenticatedError(absl::StrCat(arg1, arg2, arg3));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4) {
  return absl::UnauthenticatedError(absl::StrCat(arg1, arg2, arg3, arg4));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4,
                                    const absl::AlphaNum& arg5) {
  return absl::UnauthenticatedError(absl::StrCat(arg1, arg2, arg3, arg4, arg5));
}
ABSL_DEPRECATE_AND_INLINE()
inline absl::Status Unauthenticated(const absl::AlphaNum& arg1,
                                    const absl::AlphaNum& arg2,
                                    const absl::AlphaNum& arg3,
                                    const absl::AlphaNum& arg4,
                                    const absl::AlphaNum& arg5,
                                    const absl::AlphaNum& arg6) {
  return absl::UnauthenticatedError(
      absl::StrCat(arg1, arg2, arg3, arg4, arg5, arg6));
}
template <typename... Args>
ABSL_DEPRECATED(
    "Use absl::UnauthenticatedError(absl::StrCat(args...)) instead.")
inline ABSL_STATUS Unauthenticated(Args... rest) {
  return absl::UnauthenticatedError(::tsl::strings::StrCat(
      ::tsl::errors::internal::PrepareForStrCat(rest)...));
}
inline absl::Status UnauthenticatedWithPayloads(
    absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  return errors::Create(absl::StatusCode::kUnauthenticated, message, payloads);
}

ABSL_DEPRECATE_AND_INLINE()
inline bool IsAborted(const absl::Status& status) {
  return absl::IsAborted(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsAlreadyExists(const absl::Status& status) {
  return absl::IsAlreadyExists(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsCancelled(const absl::Status& status) {
  return absl::IsCancelled(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsDataLoss(const absl::Status& status) {
  return absl::IsDataLoss(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsDeadlineExceeded(const absl::Status& status) {
  return absl::IsDeadlineExceeded(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsFailedPrecondition(const absl::Status& status) {
  return absl::IsFailedPrecondition(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsInternal(const absl::Status& status) {
  return absl::IsInternal(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsInvalidArgument(const absl::Status& status) {
  return absl::IsInvalidArgument(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsNotFound(const absl::Status& status) {
  return absl::IsNotFound(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsOutOfRange(const absl::Status& status) {
  return absl::IsOutOfRange(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsPermissionDenied(const absl::Status& status) {
  return absl::IsPermissionDenied(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsResourceExhausted(const absl::Status& status) {
  return absl::IsResourceExhausted(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsUnauthenticated(const absl::Status& status) {
  return absl::IsUnauthenticated(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsUnavailable(const absl::Status& status) {
  return absl::IsUnavailable(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsUnimplemented(const absl::Status& status) {
  return absl::IsUnimplemented(status);
}
ABSL_DEPRECATE_AND_INLINE()
inline bool IsUnknown(const absl::Status& status) {
  return absl::IsUnknown(status);
}

// Produces a formatted string pattern from the name which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <name>}}
// Note: The pattern below determines the regex _NODEDEF_NAME_RE in the file
// tensorflow/python/client/session.py
// LINT.IfChange
inline std::string FormatNodeNameForError(absl::string_view name) {
  return absl::StrCat("{{node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/client/session.py)
template <typename T>
std::string FormatNodeNamesForError(const T& names) {
  return absl::StrJoin(names, ", ",
                       [](std::string* output, absl::string_view s) {
                         absl::StrAppend(output, FormatNodeNameForError(s));
                       });
}
// LINT.IfChange
inline std::string FormatColocationNodeForError(absl::string_view name) {
  return absl::StrCat("{{colocation_node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/framework/error_interpolation.py)
template <typename T, typename = std::enable_if_t<
                          !std::is_convertible_v<T, absl::string_view>>>
std::string FormatColocationNodeForError(const T& names) {
  return absl::StrJoin(
      names, ", ", [](std::string* output, absl::string_view s) {
        absl::StrAppend(output, FormatColocationNodeForError(s));
      });
}

inline std::string FormatFunctionForError(absl::string_view name) {
  return absl::StrCat("{{function_node ", name, "}}");
}

inline absl::Status ReplaceErrorFromNonCommunicationOps(
    const absl::Status s, absl::string_view op_name) {
  assert(absl::IsUnavailable(s));
  return absl::InternalError(absl::StrCat(
      s.message(), "\nExecuting non-communication op <", op_name,
      "> originally returned UnavailableError, and was replaced by "
      "InternalError to avoid invoking TF network error handling logic."));
}

template <typename T>
std::string FormatOriginalNodeLocationForError(const T& node_names,
                                               const T& func_names) {
  std::vector<std::string> error_message;
  for (int i = 0; i != node_names.size(); ++i) {
    if (i != 0) {
      error_message.push_back(", ");
    }
    if (i < func_names.size()) {
      error_message.push_back(FormatFunctionForError(func_names[i]));
    }
    error_message.push_back(FormatNodeNameForError(node_names[i]));
  }
  return absl::StrJoin(error_message, "");
}

// The CanonicalCode() for non-errors.
using ::tsl::error::OK;  // NOLINT

#if defined(ABSL_REFACTOR_INLINE_DEFINED_LOCALLY)
#undef ABSL_REFACTOR_INLINE
#undef ABSL_REFACTOR_INLINE_DEFINED_LOCALLY
#endif

}  // namespace errors
}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_ERRORS_H_
