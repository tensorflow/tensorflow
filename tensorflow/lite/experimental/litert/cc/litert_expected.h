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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_

#include <variant>

#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

// TODO rename this class and update it so it has the same interface
// as std::expected.
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
    ABSL_CHECK(HasValue());
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
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_
