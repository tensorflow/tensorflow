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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_SUPPORT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_SUPPORT_H_

#include <stdio.h>

#include <cstdint>
#include <iostream>  // IWYU pragma: keep
#include <memory>
#include <variant>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"  // IWYU pragma: keep
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"  // IWYU pragma: export
#include "tensorflow/lite/experimental/lrt/vendors/c/litert_compiler_plugin.h"

// Flatbuffer's raw char type.
typedef uint8_t FbCharT;

// Const view of flatbuffer's raw buffer type.
typedef absl::Span<const FbCharT> FbConstBufferT;

// Mutable view of flatbuffer's raw buffer type.
typedef absl::Span<FbCharT> FbBufferT;

// Convenience method to get raw string view from native flatbuffer buffer.
inline absl::string_view FbBufToStr(FbConstBufferT fb_buf) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

// Mutable version of above.
inline absl::string_view FbBufToStr(FbBufferT fb_buf) {
  auto fb_buf_raw = reinterpret_cast<char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

#define _CONCAT_NAME_IMPL(x, y) x##y

#define _CONCAT_NAME(x, y) _CONCAT_NAME_IMPL(x, y)

#define _RETURN_VAL(val) return val

// TODO: b/365295276 - Put all smart pointer wrappers in support.h.
struct LiteRtCompilerPluginDeleter {
  void operator()(LiteRtCompilerPlugin plugin) {
    if (plugin != nullptr) {
      LiteRtPluginDestroy(plugin);
    }
  }
};

using UniqueLiteRtCompilerPlugin =
    std::unique_ptr<LiteRtCompilerPluginT, LiteRtCompilerPluginDeleter>;

// `StatusOr` analog for litert. Very basic currently.
// TODO: b/365295276 - Figure out how to better infer template param
// and not require passing typing to macros.
template <typename T>
class LiteRtResult {
 public:
  // TODO: b/365295276 - Implement emplace for LiteRtResult.

  static LiteRtResult<T> FromValue(const T& value) {
    LiteRtResult<T> result;
    result.data_ = value;
    return result;
  }

  static LiteRtResult<T> TakeValue(T&& value) {
    LiteRtResult<T> result;
    result.data_ = std::move(value);
    return result;
  }

  static LiteRtResult<T> FromStatus(LiteRtStatus status) {
    LiteRtResult<T> result;
    result.data_ = status;
    return result;
  }

  T& Value() {
    if (!HasValue()) {
      LITERT_FATAL("Result does not contain a value.");
    }
    return std::get<T>(data_);
  }

  LiteRtStatus Status() {
    if (std::holds_alternative<T>(data_)) {
      return kLiteRtStatusOk;
    }
    return std::get<LiteRtStatus>(data_);
  }

  bool HasValue() { return std::holds_alternative<T>(data_); }

 private:
  std::variant<LiteRtStatus, T> data_;
};

#ifdef NDEBUG
#define _LITERT_D_MSG(msg)
#else
#define _LITERT_D_MSG(msg) \
  std::cerr << msg << " " << __FILE__ << ":" << __LINE__ << "\n";
#endif

#ifdef LITERT_RETURN_STATUS_IF_NOT_OK_MSG
#undef LITERT_RETURN_STATUS_IF_NOT_OK_MSG
#define LITERT_RETURN_STATUS_IF_NOT_OK_MSG(expr, d_msg)        \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) { \
    _LITERT_D_MSG(d_msg);                                      \
    return status;                                             \
  }
#endif

// TODO: b/365295276 Make c friendly `CHECK` macro(s) and move to c api.
#define LITERT_CHECK_STATUS_HAS_CODE_MSG(expr, code, d_msg) \
  if (LiteRtStatus status = expr; status != code) {         \
    _LITERT_D_MSG(d_msg);                                   \
    ABSL_CHECK(false);                                      \
  }

#define LITERT_CHECK_STATUS_HAS_CODE(expr, code) \
  LITERT_CHECK_STATUS_HAS_CODE_MSG(expr, code, "");

#define LITERT_CHECK_STATUS_OK(expr) \
  LITERT_CHECK_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

#define LITERT_CHECK_STATUS_OK_MSG(expr, d_msg) \
  LITERT_CHECK_STATUS_HAS_CODE_MSG(expr, kLiteRtStatusOk, d_msg);

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

#define LITERT_ENSURE_SUPPORTED(cond, msg)                          \
  if (!(cond)) {                                                    \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << "\n"; \
    return kLiteRtStatusErrorUnsupported;                           \
  }

#define LITERT_ENSURE(expr, fail_stat, msg)                         \
  if (!(expr)) {                                                    \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << "\n"; \
    return fail_stat;                                               \
  }

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_SUPPORT_H_
