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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_

#include <any>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert {

inline std::any ToStdAny(LiteRtAny litert_any) {
  std::any res;
  switch (litert_any.type) {
    case kLiteRtAnyTypeNone:
      break;
    case kLiteRtAnyTypeBool:
      res = litert_any.bool_value;
      break;
    case kLiteRtAnyTypeInt:
      res = litert_any.int_value;
      break;
    case kLiteRtAnyTypeReal:
      res = litert_any.real_value;
      break;
    case kLiteRtAnyTypeString:
      res = litert_any.str_value;
      break;
    case kLiteRtAnyTypeVoidPtr:
      res = litert_any.ptr_value;
      break;
  }
  return res;
}

inline Expected<LiteRtAny> ToLiteRtAny(const std::any& any) {
  LiteRtAny result;
  if (!any.has_value()) {
    result.type = kLiteRtAnyTypeNone;
    return result;

  } else if (any.type() == typeid(LiteRtAny::bool_value)) {
    result.type = kLiteRtAnyTypeBool;
    result.bool_value = std::any_cast<decltype(LiteRtAny::bool_value)>(any);
    return result;

  } else if (any.type() == typeid(int8_t)) {
    result.type = kLiteRtAnyTypeInt;
    result.int_value = std::any_cast<int8_t>(any);
    return result;

  } else if (any.type() == typeid(int16_t)) {
    result.type = kLiteRtAnyTypeInt;
    result.int_value = std::any_cast<int16_t>(any);
    return result;

  } else if (any.type() == typeid(int32_t)) {
    result.type = kLiteRtAnyTypeInt;
    result.int_value = std::any_cast<int32_t>(any);
    return result;

  } else if (any.type() == typeid(int64_t)) {
    result.type = kLiteRtAnyTypeInt;
    result.int_value = std::any_cast<int64_t>(any);
    return result;

  } else if (any.type() == typeid(float)) {
    result.type = kLiteRtAnyTypeReal;
    result.real_value = std::any_cast<float>(any);
    return result;

  } else if (any.type() == typeid(double)) {
    result.type = kLiteRtAnyTypeReal;
    result.real_value = std::any_cast<double>(any);
    return result;

  } else if (any.type() == typeid(LiteRtAny::str_value)) {
    result.type = kLiteRtAnyTypeString;
    result.str_value = std::any_cast<decltype(LiteRtAny::str_value)>(any);
    return result;

  } else if (any.type() == typeid(absl::string_view)) {
    result.type = kLiteRtAnyTypeString;
    result.str_value = std::any_cast<absl::string_view>(any).data();
    return result;

  } else if (any.type() == typeid(LiteRtAny::ptr_value)) {
    result.type = kLiteRtAnyTypeVoidPtr;
    result.ptr_value = std::any_cast<decltype(LiteRtAny::ptr_value)>(any);
    return result;

  } else {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid argument for ToLiteRtAny");
  }
}

namespace internal {

inline Expected<void> CheckType(const LiteRtAny& any,
                                const LiteRtAnyType type) {
  if (any.type != kLiteRtAnyTypeString) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Wrong LiteRtAny type. Expected %s, got %s.",
                                 LiteRtAnyTypeToString(type),
                                 LiteRtAnyTypeToString(any.type)));
  }
  return {};
}

template <class T>
Expected<T> GetInt(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(CheckType(any, kLiteRtAnyTypeInt));
  if (any.int_value > std::numeric_limits<T>::max() ||
      any.int_value < std::numeric_limits<T>::lowest()) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("LiteRtAny integer is out of range. %v <= %v <= %v",
                        std::numeric_limits<T>::lowest(), any.int_value,
                        std::numeric_limits<T>::max()));
  }
  return any.int_value;
}

template <class T>
Expected<T> GetReal(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(CheckType(any, kLiteRtAnyTypeReal));
  if (any.real_value > std::numeric_limits<T>::max() ||
      any.real_value < std::numeric_limits<T>::lowest()) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat(
            "LiteRtAny integer is out of range. %v <= %v <= %v failed.",
            std::numeric_limits<T>::lowest(), any.real_value,
            std::numeric_limits<T>::max()));
  }
  return any.real_value;
}
}  // namespace internal

// Extracts the value from a LiteRtAny object with type checking.
template <class T>
inline Expected<T> Get(const LiteRtAny& any);

template <>
inline Expected<bool> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeBool));
  return any.bool_value;
}

template <>
inline Expected<int8_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int8_t>(any);
}

template <>
inline Expected<int16_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int16_t>(any);
}

template <>
inline Expected<int32_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int32_t>(any);
}

template <>
inline Expected<int64_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int64_t>(any);
}

template <>
inline Expected<float> Get(const LiteRtAny& any) {
  return internal::GetReal<float>(any);
}

template <>
inline Expected<double> Get(const LiteRtAny& any) {
  return internal::GetReal<double>(any);
}

template <>
inline Expected<std::string> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeString));
  return std::string(any.str_value);
}

template <>
inline Expected<absl::string_view> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeString));
  return absl::string_view(any.str_value);
}

template <>
inline Expected<const void*> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeVoidPtr));
  return any.ptr_value;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_
