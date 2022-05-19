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

#ifndef TENSORFLOW_CORE_PLATFORM_ERRORS_H_
#define TENSORFLOW_CORE_PLATFORM_ERRORS_H_

#include <sstream>
#include <string>
#include <utility>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
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
typename std::enable_if<!std::is_convertible<T, strings::AlphaNum>::value,
                        std::string>::type
PrepareForStrCat(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}
inline const strings::AlphaNum& PrepareForStrCat(const strings::AlphaNum& a) {
  return a;
}

}  // namespace internal

// Maps UNIX errors into a Status.
Status IOError(const string& context, int err_number);

// Returns all payloads from a Status as a key-value map.
inline std::unordered_map<std::string, std::string> GetPayloads(
    const ::tensorflow::Status& status) {
  std::unordered_map<std::string, std::string> payloads;
  status.ForEachPayload(
      [&payloads](tensorflow::StringPiece key, tensorflow::StringPiece value) {
        payloads[std::string(key)] = std::string(value);
      });
  return payloads;
}

// Inserts all given payloads into the given status. Will overwrite existing
// payloads if they exist with the same key.
inline void InsertPayloads(
    ::tensorflow::Status& status,
    const std::unordered_map<std::string, std::string>& payloads) {
  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, payload.second);
  }
}

// Copies all payloads from one Status to another. Will overwrite existing
// payloads in the destination if they exist with the same key.
inline void CopyPayloads(const ::tensorflow::Status& from,
                         ::tensorflow::Status& to) {
  from.ForEachPayload(
      [&to](tensorflow::StringPiece key, tensorflow::StringPiece value) {
        to.SetPayload(key, value);
      });
}

// Creates a new status with the given code, message and payloads.
inline ::tensorflow::Status Create(
    Code code, ::tensorflow::StringPiece message,
    const std::unordered_map<std::string, std::string>& payloads) {
  Status status(code, message);
  InsertPayloads(status, payloads);
  return status;
}

// Returns a new Status, replacing its message with the given.
inline ::tensorflow::Status CreateWithUpdatedMessage(
    const ::tensorflow::Status& status, ::tensorflow::StringPiece message) {
  return Create(status.code(), message, GetPayloads(status));
}

// Append some context to an error message.  Each time we append
// context put it on a new line, since it is possible for there
// to be several layers of additional context.
template <typename... Args>
void AppendToMessage(::tensorflow::Status* status, Args... args) {
  std::vector<StackFrame> stack_trace = status->stack_trace();
  auto new_status = ::tensorflow::Status(
      status->code(),
      ::tensorflow::strings::StrCat(status->error_message(), "\n\t", args...),
      std::move(stack_trace));
  CopyPayloads(*status, new_status);
  *status = std::move(new_status);
}

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                          \
  do {                                                   \
    ::tensorflow::Status _status = (__VA_ARGS__);        \
    if (TF_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define TF_RETURN_WITH_CONTEXT_IF_ERROR(expr, ...)                  \
  do {                                                              \
    ::tensorflow::Status _status = (expr);                          \
    if (TF_PREDICT_FALSE(!_status.ok())) {                          \
      ::tensorflow::errors::AppendToMessage(&_status, __VA_ARGS__); \
      return _status;                                               \
    }                                                               \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CONST)                                        \
  template <typename... Args>                                             \
  ::tensorflow::Status FUNC(Args... args) {                               \
    return ::tensorflow::Status(                                          \
        ::tensorflow::error::CONST,                                       \
        ::tensorflow::strings::StrCat(                                    \
            ::tensorflow::errors::internal::PrepareForStrCat(args)...));  \
  }                                                                       \
  template <typename... Args>                                             \
  ::tensorflow::Status FUNC##WithPayloads(                                \
      const ::tensorflow::StringPiece& message,                           \
      const std::unordered_map<std::string, std::string>& payloads) {     \
    return errors::Create(::tensorflow::error::CONST, message, payloads); \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

bool IsAborted(const Status& status);
bool IsAlreadyExists(const Status& status);
bool IsCancelled(const Status& status);
bool IsDataLoss(const Status& status);
bool IsDeadlineExceeded(const Status& status);
bool IsFailedPrecondition(const Status& status);
bool IsInternal(const Status& status);
bool IsInvalidArgument(const Status& status);
bool IsNotFound(const Status& status);
bool IsOutOfRange(const Status& status);
bool IsPermissionDenied(const Status& status);
bool IsResourceExhausted(const Status& status);
bool IsUnauthenticated(const Status& status);
bool IsUnavailable(const Status& status);
bool IsUnimplemented(const Status& status);
bool IsUnknown(const Status& status);

#undef DECLARE_ERROR

// Produces a formatted string pattern from the name which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <name>}}
// Note: The pattern below determines the regex _NODEDEF_NAME_RE in the file
// tensorflow/python/client/session.py
// LINT.IfChange
inline std::string FormatNodeNameForError(const std::string& name) {
  return strings::StrCat("{{node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/client/session.py)
template <typename T>
std::string FormatNodeNamesForError(const T& names) {
  return absl::StrJoin(
      names, ", ", [](std::string* output, const std::string& s) {
        ::tensorflow::strings::StrAppend(output, FormatNodeNameForError(s));
      });
}
// LINT.IfChange
inline std::string FormatColocationNodeForError(const std::string& name) {
  return strings::StrCat("{{colocation_node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/framework/error_interpolation.py)
template <typename T>
std::string FormatColocationNodeForError(const T& names) {
  return absl::StrJoin(names, ", ",
                       [](std::string* output, const std::string& s) {
                         ::tensorflow::strings::StrAppend(
                             output, FormatColocationNodeForError(s));
                       });
}

inline std::string FormatFunctionForError(const std::string& name) {
  return strings::StrCat("{{function_node ", name, "}}");
}

inline Status ReplaceErrorFromNonCommunicationOps(const Status s,
                                                  const std::string& op_name) {
  assert(tensorflow::IsUnavailable(s));
  return Status(
      error::INTERNAL,
      strings::StrCat(
          s.error_message(), "\nExecuting non-communication op <", op_name,
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
using ::tensorflow::error::OK;

}  // namespace errors
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ERRORS_H_
