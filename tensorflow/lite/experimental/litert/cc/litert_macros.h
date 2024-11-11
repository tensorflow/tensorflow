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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"


#define _CONCAT_NAME_IMPL(x, y) x##y

#define _CONCAT_NAME(x, y) _CONCAT_NAME_IMPL(x, y)

#define _RETURN_VAL(val) return val

#define LITERT_CHECK_STATUS_HAS_CODE(expr, code) ABSL_CHECK(expr == code);

#define LITERT_CHECK_STATUS_OK(expr) \
  LITERT_CHECK_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

// If expr doesn't retur ok status, wrap as result and return.
#define LITERT_RETURN_RESULT_IF_NOT_OK(expr, ty)             \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) \
    return LiteRtResult<ty>::FromStatus(status);

#define _ASSIGN_OR_BLOCK(decl, expr, block, result) \
  auto result = (expr);                             \
  if (!result.HasValue()) {                         \
    block;                                          \
  }                                                 \
  decl = result.Value();

#define _MOVE_OR_BLOCK(decl, expr, block, result) \
  auto result = (expr);                           \
  if (!result.HasValue()) {                       \
    block;                                        \
  }                                               \
  decl = std::move(result.Value());

#define _MOVE_OR_RETURN_VAL(decl, expr, val, result) \
  _MOVE_OR_BLOCK(decl, expr, _RETURN_VAL(val), result)

#define _ASSIGN_OR_RETURN_VAL(decl, expr, val, result) \
  _ASSIGN_OR_BLOCK(decl, expr, _RETURN_VAL(val), result)

// Assign value behind result returned from expr. If not ok, return val.
#define LITERT_ASSIGN_OR_RETURN_VAL(decl, expr, val) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, val, _CONCAT_NAME(_result, __COUNTER__))

#define _STATUS_FROM_RESULT(result) result.Status();

#define _ASSIGN_OR_RETURN_STATUS(decl, expr, result) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, _STATUS_FROM_RESULT(result), result)

#define _MOVE_OR_RETURN_STATUS(decl, expr, result) \
  _MOVE_OR_RETURN_VAL(decl, expr, _STATUS_FROM_RESULT(result), result)

// Assign value behind result returned from expr. If not ok, return status.
#define LITERT_ASSIGN_OR_RETURN_STATUS(decl, expr) \
  _ASSIGN_OR_RETURN_STATUS(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

// Assign value behind result returned from expr. If not ok, return status.
#define LITERT_MOVE_OR_RETURN_STATUS(decl, expr) \
  _MOVE_OR_RETURN_STATUS(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define _FORWARD_RESULT(result, ty) \
  LiteRtResult<ty>::FromStatus(result.Status());

#define _ASSIGN_OR_RETURN_RESULT(decl, expr, ty, result) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, _FORWARD_RESULT(result, ty), result)

// Assign value behind result returned from expr. If not ok, return result.
#define LITERT_ASSIGN_OR_RETURN_RESULT(decl, expr, ty) \
  _ASSIGN_OR_RETURN_RESULT(decl, expr, ty, _CONCAT_NAME(_result, __COUNTER__))

#define _MOVE_OR_RETURN_RESULT(decl, expr, ty, result) \
  _MOVE_OR_RETURN_VAL(decl, expr, _FORWARD_RESULT(result, ty), result)

// Move value behind result returned from expr. If not ok, return result.
#define LITERT_MOVE_OR_RETURN_RESULT(decl, expr, ty) \
  _MOVE_OR_RETURN_RESULT(decl, expr, ty, _CONCAT_NAME(_result, __COUNTER__))

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

#define LITERT_RETURN_STATUS_IF_NOT_OK(expr) \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) return status;

#define LITERT_RETURN_STATUS_IF_NOT_OK_OR_NOT_MATCHED(expr)                  \
  if (LiteRtStatus status = expr;                                            \
      (status != kLiteRtStatusOk && status != kLiteRtStatusLegalizeNoMatch)) \
    return status;

#define LITERT_RETURN_VAL_IF_NOT_OK(expr, ret_val) \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) return ret_val;

#define LITERT_STACK_ARRAY(ty, var, size, init) \
  ty* var = (ty*)alloca(sizeof(ty) * size);     \
  for (ty* e = var; e < var + size; ++e) {      \
    *e = init;                                  \
  }


#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MACROS_H_
