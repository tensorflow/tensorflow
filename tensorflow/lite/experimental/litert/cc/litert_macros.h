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

#include "absl/log/absl_check.h"
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
// `litert::Expected` holding an error or a `LiteRtStatus`).
//
// Returns `return_expr` if the result of `expr` represents an error.
//
// The result of `expr` may be referenced as `status` in `return_expr`.
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
#define LITERT_ASSIGN_OR_RETURN(DECL, ...)                                     \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD((DECL, __VA_ARGS__,                  \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_3,   \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_2))( \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

//////////// Implementation details start here. ///////////////////////

// Converts implicitly to either `LiteRtStatus` or `litert::Expected` holding an
// error. This allows returning a status in functions using either of these as a
// return type in `LITERT_RETURN_IF_ERROR`.
class ErrorStatusReturnHelper {
 public:
  template <class T>
  explicit ErrorStatusReturnHelper(const litert::Expected<T>& expected)
      : error_(expected.Error()) {}
  explicit ErrorStatusReturnHelper(LiteRtStatus status) : error_(status) {}
  explicit ErrorStatusReturnHelper(const litert::Unexpected& unexpected)
      : error_(unexpected.Error()) {}

  // NOLINTBEGIN(*-explicit-constructor): This class transparently converts to
  // `LiteRtStatus` and `litert::Exepected`.
  operator LiteRtStatus() const noexcept { return error_.Status(); }

  template <class T>
  operator litert::Expected<T>() const noexcept {
    return litert::Unexpected(error_);
  }
  // NOLINTEND(*-explicit-constructor)

  static constexpr bool IsError(LiteRtStatus status) {
    return status != kLiteRtStatusOk;
  }

  static constexpr bool IsError(const litert::Unexpected&) { return true; }

  template <class T>
  static constexpr bool IsError(const litert::Expected<T>& expected) {
    return !expected.HasValue();
  }

 private:
  litert::Error error_;
};

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER(_1, _2, OVERLOAD, ...) \
  OVERLOAD

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD(args) \
  LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER args

#define LITERT_RETURN_IF_ERROR_1(EXPR) \
  LITERT_RETURN_IF_ERROR_2(EXPR, ErrorStatusReturnHelper{status})

#define LITERT_RETURN_IF_ERROR_2(EXPR, RETURN_VALUE)                      \
  do {                                                                    \
    if (auto status = (EXPR); ErrorStatusReturnHelper::IsError(status)) { \
      return RETURN_VALUE;                                                \
    }                                                                     \
  } while (false)

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER(_1, _2, _3, OVERLOAD, \
                                                       ...)                  \
  OVERLOAD

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD(args) \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER args

#define LITERT_ASSIGN_OR_RETURN_HELPER_2(TMP_VAR, DECL, EXPR) \
  LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR,       \
                                   ErrorStatusReturnHelper(TMP_VAR))

#define LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, RETURN_VALUE) \
  auto&& TMP_VAR = (EXPR);                                                  \
  if (ErrorStatusReturnHelper::IsError(TMP_VAR)) {                          \
    return RETURN_VALUE;                                                    \
  }                                                                         \
  DECL = std::move(TMP_VAR.Value());

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MACROS_H_
