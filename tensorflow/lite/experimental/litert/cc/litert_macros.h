// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MACROS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MACROS_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"  // IWYU pragma: keep

#define _CONCAT_NAME_IMPL(x, y) x##y

#define _CONCAT_NAME(x, y) _CONCAT_NAME_IMPL(x, y)

#define _RETURN_VAL(val) return val

#define LITERT_CHECK_STATUS_HAS_CODE(expr, code) ABSL_CHECK(expr == code);

#define LITERT_CHECK_STATUS_OK(expr) \
  LITERT_CHECK_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

#define LITERT_ENSURE_SUPPORTED(cond, msg) \
  if (!(cond)) {                           \
    LITERT_LOG(LITERT_ERROR, "%s", msg);   \
    return kLiteRtStatusErrorUnsupported;  \
  }

#define LITERT_ENSURE(expr, fail_stat, msg) \
  if (!(expr)) {                            \
    LITERT_LOG(LITERT_ERROR, "%s", msg);    \
    return fail_stat;                       \
  }

#define LITERT_RETURN_IF_ERROR_OR_NOT_MATCHED(expr)                          \
  if (LiteRtStatus status = expr;                                            \
      (status != kLiteRtStatusOk && status != kLiteRtStatusLegalizeNoMatch)) \
    return status;

#define LITERT_STACK_ARRAY(ty, var, size, init) \
  ty* var = (ty*)alloca(sizeof(ty) * size);     \
  for (ty* e = var; e < var + size; ++e) {      \
    *e = init;                                  \
  }

// LITERT_RETURN_IF_ERROR(expr);
// LITERT_RETURN_IF_ERROR(expr, return_value);
//
// Returns the result of `expr` if it represents an LiteRT error status (either
// `litert::Expected` holding an error, a `LiteRtStatus` or a bool that
// evaluated to `false`).
//
// Returns `return_value` if the result of `expr` represents an error.
//
// The result of `expr` may be referenced as `status` in `return_expr`.
//
// By default, the return value is an `ErrorStatusBuilder` built from using the
// result of `expr`. The error message of this builder can be customized by
// using its `*Log*()` functions and the << operator.
//
// ```cpp
// LITERT_RETURN_IF_ERROR(expr) << "Failed while trying to ...";
// ```
#define LITERT_RETURN_IF_ERROR(...)                                       \
  LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD(                                 \
      (__VA_ARGS__, LITERT_RETURN_IF_ERROR_2, LITERT_RETURN_IF_ERROR_1))( \
      __VA_ARGS__)

// ASSIGN_OR_RETURN(decl, expr)
// ASSIGN_OR_RETURN(decl, expr, return_value)
//
// Evaluates `expr` that should convert to a `litert::Expected` object.
//
// - If the object holds a value, move-assigns the value to `decl`.
// - If the object holds an error, returns the error, casting it to a
// `LiteRtStatus` if required.
//
// `return_value` may be specified to return a custom value in case of error.
//
// By when specifying `return_value`, an `ErrorStatusBuilder` variable called
// `_` can be used to customize the error message.
//
// ```cpp
// LITERT_ASSIGN_OR_RETURN(expr, _ << "Failed while trying to ...");
// ```
#define LITERT_ASSIGN_OR_RETURN(DECL, ...)                                     \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD((DECL, __VA_ARGS__,                  \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_3,   \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_2))( \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

namespace litert {

#if defined(__has_builtin) && __has_builtin(__builtin_FILE) && \
    __has_builtin(__builtin_LINE)
#define LITERT_INTERNAL_BUILTIN_FILE __builtin_FILE()
#define LITERT_INTERNAL_BUILTIN_LINE __builtin_LINE()
#else
#define LITERT_INTERNAL_BUILTIN_FILE "unknown"
#define LITERT_INTERNAL_BUILTIN_LINE 0
#endif

// Stores a file and a line number.
//
// Mimics a subset of `std::source_location` to be replaced by it when we update
// to C++20.
class SourceLocation {
  // We have this to prevent `current()` parameters from begin modified.
  struct PrivateTag {};

 public:
  // Creates a SourceLocation with the line and file corresponding to the
  // call site.
  static constexpr SourceLocation current(
      PrivateTag = PrivateTag{},
      const char* file = LITERT_INTERNAL_BUILTIN_FILE,
      uint32_t line = LITERT_INTERNAL_BUILTIN_LINE) {
    return SourceLocation{file, line};
  }

  constexpr const char* file_name() const { return file_; }
  constexpr uint32_t line() const { return line_; }

 private:
  // Builds a SourceLocation object.
  //
  // Note: This is private as `std::source_location` doesn't provide a way of
  // manually building a source location.
  constexpr SourceLocation(const char* file, uint32_t line)
      : file_(file), line_(line) {}

  const char* file_;
  uint32_t line_;
};

// Converts implicitly to either `LiteRtStatus` or `litert::Expected` holding an
// error. This allows returning a status in functions using either of these as a
// return type in `LITERT_RETURN_IF_ERROR` and `LITERT_ASSIGN_OR_RETURN`.
//
// When a C++ error with a message is converted to a `LiteRtStatus`, the message
// is logged (as an error by default, use the `Log*()` functions to customize
// that).
//
// The error message may be completed with extra info by using the << operator.
class ErrorStatusBuilder {
 public:
  explicit ErrorStatusBuilder(
      bool expr_result,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(kLiteRtStatusErrorUnknown), loc_(loc) {}

  template <class T>
  explicit ErrorStatusBuilder(
      const litert::Expected<T>& expected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(expected.Error()), loc_(loc) {}

  template <class T>
  explicit ErrorStatusBuilder(
      litert::Expected<T>&& expected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(std::move(expected.Error())), loc_(loc) {}

  explicit ErrorStatusBuilder(
      LiteRtStatus status,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(status), loc_(loc) {}

  explicit ErrorStatusBuilder(
      const litert::Unexpected& unexpected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(unexpected.Error()), loc_(loc) {}

  explicit ErrorStatusBuilder(
      litert::Unexpected&& unexpected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(std::move(unexpected.Error())), loc_(loc) {}

  // NOLINTBEGIN(*-explicit-constructor): This class transparently converts to
  // `LiteRtStatus` and `litert::Expected`.

  // Note: this conversion logs the error message if there is one unless NDEBUG
  // is set (generally in case of optimized builds).
  operator LiteRtStatus() const noexcept {
#ifndef NDEBUG
    if (ShouldLog()) {
      LiteRtLogger logger = LiteRtGetDefaultLogger();
      LiteRtLogSeverity __min_severity__;
      if (LiteRtGetMinLoggerSeverity(logger, &__min_severity__) !=
          kLiteRtStatusOk) {
        __min_severity__ = kLiteRtLogSeverityVerbose;
      }
      if (log_level_ >= __min_severity__) {
        LiteRtLoggerLog(logger, log_level_, "[%s:%u] %s", loc_.file_name(),
                        loc_.line(), LogMessage().c_str());
      }
    }
#endif
    return error_.Status();
  }

  template <class T>
  operator litert::Expected<T>() const noexcept {
    return litert::Unexpected(error_.Status(), LogMessage());
  }

  operator absl::Status() const noexcept;

  template <class T>
  operator absl::StatusOr<T>() const noexcept {
    return static_cast<absl::Status>(*this);
  }
  // NOLINTEND(*-explicit-constructor)

  static constexpr bool IsError(bool status) { return !status; }

  static constexpr bool IsError(LiteRtStatus status) {
    return status != kLiteRtStatusOk;
  }

  static constexpr bool IsError(const litert::Unexpected&) { return true; }

  template <class T>
  static constexpr bool IsError(const litert::Expected<T>& expected) {
    return !expected.HasValue();
  }

  // Appends data to the error message.
  template <class T>
  ErrorStatusBuilder& operator<<(T&& val) {
    if (!extra_log_) {
      extra_log_ = std::make_unique<std::stringstream>();
    }
    *extra_log_ << static_cast<T&&>(val);
    return *this;
  }

  // Sets the log level used when converting to a `LiteRtStatus`.
  ErrorStatusBuilder& Log(LiteRtLogSeverity log_level) noexcept {
    log_level_ = log_level;
    return *this;
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogVerbose() noexcept {
    return Log(kLiteRtLogSeverityVerbose);
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `info`.
  ErrorStatusBuilder& LogInfo() noexcept { return Log(kLiteRtLogSeverityInfo); }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogWarning() noexcept {
    return Log(kLiteRtLogSeverityWarning);
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogError() noexcept {
    return Log(kLiteRtLogSeverityError);
  }

  // Prevent logging any message when converting to a `LiteRtStatus`.
  ErrorStatusBuilder& NoLog() noexcept { return Log(kLiteRtLogSeveritySilent); }

 private:
  bool ShouldLog() const noexcept {
    return log_level_ != kLiteRtLogSeveritySilent &&
           (!error_.Message().empty() || extra_log_);
  }

  std::string LogMessage() const {
    if (!error_.Message().empty() && extra_log_) {
      std::string res;
      res.reserve(error_.Message().size() + extra_log_->tellp() + 2);
      res.append(error_.Message());
      res.append(" ");
      res.append(extra_log_->str());
      return res;
    }
    if (!error_.Message().empty()) {
      return error_.Message();
    }
    if (extra_log_) {
      return extra_log_->str();
    }
    return {};
  }

  litert::Error error_;
  litert::SourceLocation loc_;
  std::unique_ptr<std::stringstream> extra_log_;
  LiteRtLogSeverity log_level_ = kLiteRtLogSeverityError;
};

}  // namespace litert

//////////// Implementation details start here. ///////////////////////

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER(_1, _2, OVERLOAD, ...) \
  OVERLOAD

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD(args) \
  LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER args

#define LITERT_RETURN_IF_ERROR_1(EXPR) \
  LITERT_RETURN_IF_ERROR_2(EXPR,       \
                           ::litert::ErrorStatusBuilder{std::move(status)})

#define LITERT_RETURN_IF_ERROR_2(EXPR, RETURN_VALUE)                       \
  if (auto status = (EXPR); ::litert::ErrorStatusBuilder::IsError(status)) \
  return RETURN_VALUE

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER(_1, _2, _3, OVERLOAD, \
                                                       ...)                  \
  OVERLOAD

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD(args) \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER args

#define LITERT_ASSIGN_OR_RETURN_HELPER_2(TMP_VAR, DECL, EXPR) \
  LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, RETURN_VALUE) \
  auto&& TMP_VAR = (EXPR);                                                  \
  if (::litert::ErrorStatusBuilder::IsError(TMP_VAR)) {                     \
    [[maybe_unused]] ::litert::ErrorStatusBuilder _(std::move(TMP_VAR));    \
    return RETURN_VALUE;                                                    \
  }                                                                         \
  DECL = std::move(TMP_VAR.Value());

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MACROS_H_
