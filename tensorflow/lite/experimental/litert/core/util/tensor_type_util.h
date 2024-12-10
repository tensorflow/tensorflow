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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

struct Ratio {
  using Type = int;
  Type num;
  Type denom;
  std::string ToString() const { return absl::StrCat(num, "/", denom); }
};

Expected<Ratio> GetElementSize(LiteRtElementType element_type);

// Get the number of elements in a tensor with given dimensions.
template <typename T>
Expected<size_t> GetNumElements(absl::Span<T> dimensions) {
  size_t num_elements = 1;
  for (auto i = 0; i < dimensions.size(); ++i) {
    auto dim = dimensions[i];
    if (dim < 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected negative dimension");
    } else if (dim == 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected 0 dimension");
    }
    num_elements *= dim;
  }
  return num_elements;
}

inline Expected<size_t> GetNumElements(
    const LiteRtRankedTensorType& tensor_type) {
  return GetNumElements(
      absl::MakeSpan(tensor_type.layout.dimensions, tensor_type.layout.rank));
}

// Get the minimum number of bytes necessary to represent a packed tensor with a
// given element type and dimensions.
template <typename T>
Expected<size_t> GetNumPackedBytes(LiteRtElementType element_type,
                                   absl::Span<T> dimensions) {
  auto element_size = GetElementSize(element_type);
  if (!element_size) {
    return element_size.Error();
  }
  auto num_elements = GetNumElements(dimensions);
  if (!num_elements) {
    return num_elements.Error();
  }
  return ((*num_elements * element_size->num) + (element_size->denom - 1)) /
         element_size->denom;
}

// Get the number of bytes necessary to represent a packed tensor type, ignoring
// any stride information.
inline Expected<size_t> GetNumPackedBytes(
    const LiteRtRankedTensorType& tensor_type) {
  return GetNumPackedBytes(
      tensor_type.element_type,
      absl::MakeSpan(tensor_type.layout.dimensions, tensor_type.layout.rank));
}

// Get the minimum number of bytes necessary to represent a possibly unpacked
// tensor with a given element type, dimensions, and strides.
template <typename T, typename U>
Expected<size_t> GetNumBytes(LiteRtElementType element_type,
                             absl::Span<T> dimensions, absl::Span<U> strides) {
  if (dimensions.size() != strides.size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Dimensions and strides have different number of elements");
  }
  auto element_size = GetElementSize(element_type);
  if (!element_size) {
    return element_size.Error();
  }
  auto rank = dimensions.size();
  size_t num_elements = 1;
  for (auto i = 0; i < rank; ++i) {
    num_elements += (dimensions[i] - 1) * strides[i];
  }
  return ((num_elements * element_size->num) + (element_size->denom - 1)) /
         element_size->denom;
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_
