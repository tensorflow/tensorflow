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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_SUPPORT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_SUPPORT_H_

#include <stdio.h>

#include <iostream>  // IWYU pragma: keep
#include <memory>
#include <variant>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"  // IWYU pragma: export
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"

#define _CONCAT_NAME_IMPL(x, y) x##y

#define _CONCAT_NAME(x, y) _CONCAT_NAME_IMPL(x, y)

#define _RETURN_VAL(val) return val

// TODO: b/365295276 - Put all smart pointer wrappers in support.h.
struct LrtCompilerPluginDeleter {
  void operator()(LrtCompilerPlugin plugin) {
    if (plugin != nullptr) {
      LrtPluginDestroy(plugin);
    }
  }
};

using UniqueLrtCompilerPlugin =
    std::unique_ptr<LrtCompilerPluginT, LrtCompilerPluginDeleter>;

// `StatusOr` analog for lrt. Very basic currently.
// TODO: b/365295276 - Figure out how to better infer template param
// and not require passing typing to macros.
template <typename T>
class LrtResult {
 public:
  // TODO: b/365295276 - Implement emplace for LrtResult.

  static LrtResult<T> FromValue(const T& value) {
    LrtResult<T> result;
    result.data_ = value;
    return result;
  }

  static LrtResult<T> TakeValue(T&& value) {
    LrtResult<T> result;
    result.data_ = std::move(value);
    return result;
  }

  static LrtResult<T> FromStatus(LrtStatus status) {
    LrtResult<T> result;
    result.data_ = status;
    return result;
  }

  T& Value() {
    if (!HasValue()) {
      LRT_FATAL("Result does not contain a value.");
    }
    return std::get<T>(data_);
  }

  LrtStatus Status() {
    if (std::holds_alternative<T>(data_)) {
      return kLrtStatusOk;
    }
    return std::get<LrtStatus>(data_);
  }

  bool HasValue() { return std::holds_alternative<T>(data_); }

 private:
  std::variant<LrtStatus, T> data_;
};

#ifdef NDEBUG
#define _LRT_D_MSG(msg)
#else
#define _LRT_D_MSG(msg) \
  std::cerr << msg << " " << __FILE__ << ":" << __LINE__ << "\n";
#endif

#ifdef LRT_RETURN_STATUS_IF_NOT_OK_MSG
#undef LRT_RETURN_STATUS_IF_NOT_OK_MSG
#define LRT_RETURN_STATUS_IF_NOT_OK_MSG(expr, d_msg)     \
  if (LrtStatus status = expr; status != kLrtStatusOk) { \
    _LRT_D_MSG(d_msg);                                   \
    return status;                                       \
  }
#endif

// TODO: b/365295276 Make c friendly `CHECK` macro(s) and move to c api.
#define LRT_CHECK_STATUS_HAS_CODE_MSG(expr, code, d_msg) \
  if (LrtStatus status = expr; status != code) {         \
    _LRT_D_MSG(d_msg);                                   \
    ABSL_CHECK(false);                                   \
  }

#define LRT_CHECK_STATUS_HAS_CODE(expr, code) \
  LRT_CHECK_STATUS_HAS_CODE_MSG(expr, code, "");

#define LRT_CHECK_STATUS_OK(expr) LRT_CHECK_STATUS_HAS_CODE(expr, kLrtStatusOk);

#define LRT_CHECK_STATUS_OK_MSG(expr, d_msg) \
  LRT_CHECK_STATUS_HAS_CODE_MSG(expr, kLrtStatusOk, d_msg);

// If expr doesn't retur ok status, wrap as result and return.
#define LRT_RETURN_RESULT_IF_NOT_OK(expr, ty)          \
  if (LrtStatus status = expr; status != kLrtStatusOk) \
    return LrtResult<ty>::FromStatus(status);

#define _ASSIGN_OR_BLOCK(decl, expr, block, result) \
  auto result = (expr);                             \
  if (!result.HasValue()) {                         \
    block;                                          \
  }                                                 \
  decl = result.Value();

#define _ASSIGN_OR_RETURN_VAL(decl, expr, val, result) \
  _ASSIGN_OR_BLOCK(decl, expr, _RETURN_VAL(val), result)

// Assign value behind result returned from expr. If not ok, return val.
#define LRT_ASSIGN_OR_RETURN_VAL(decl, expr, val) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, val, _CONCAT_NAME(_result, __COUNTER__))

#define _STATUS_FROM_RESULT(result) result.Status();

#define _ASSIGN_OR_RETURN_STATUS(decl, expr, result) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, _STATUS_FROM_RESULT(result), result)

// Assign value behind result returned from expr. If not ok, return status.
#define LRT_ASSIGN_OR_RETURN_STATUS(decl, expr) \
  _ASSIGN_OR_RETURN_STATUS(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define _FORWARD_RESULT(result, ty) LrtResult<ty>::FromStatus(result.Status());

#define _ASSIGN_OR_RETURN_RESULT(decl, expr, ty, result) \
  _ASSIGN_OR_RETURN_VAL(decl, expr, _FORWARD_RESULT(result, ty), result)

// Assign value behind result returned from expr. If not ok, return result.
#define LRT_ASSIGN_OR_RETURN_RESULT(decl, expr, ty) \
  _ASSIGN_OR_RETURN_RESULT(decl, expr, ty, _CONCAT_NAME(_result, __COUNTER__))

#define LRT_ENSURE_SUPPORTED(cond, msg)                             \
  if (!(cond)) {                                                    \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << "\n"; \
    return kLrtStatusErrorUnsupported;                              \
  }

#define LRT_ENSURE(expr, fail_stat, msg)                            \
  if (!(expr)) {                                                    \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << "\n"; \
    return fail_stat;                                               \
  }

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_SUPPORT_H_
