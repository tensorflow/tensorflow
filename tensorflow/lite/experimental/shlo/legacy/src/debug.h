/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DEBUG_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DEBUG_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/bf16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/f16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

bool AlmostSame(const Tensor& x, const Tensor& y);
bool AlmostSame(const QuantizedTensor& x, const QuantizedTensor& y);

std::ostream& operator<<(std::ostream& os, ElementType element_type);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
std::ostream& operator<<(std::ostream& os, const TensorType& tensor_type);
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
std::ostream& operator<<(std::ostream& os, const QuantizedTensorElementType& t);
std::ostream& operator<<(std::ostream& os,
                         const QuantizedTensorType& tensor_type);
std::ostream& operator<<(std::ostream& os, const QuantizedTensor& tensor);

inline std::ostream& operator<<(std::ostream& os, BF16 value) {
  return os << static_cast<float>(value);
}

inline std::ostream& operator<<(std::ostream& os, F16 value) {
  return os << static_cast<float>(value);
}

std::ostream& operator<<(std::ostream& os,
                         ComparisonDirection comparison_direction);
std::ostream& operator<<(std::ostream& os, CompareType compare_type);

std::ostream& operator<<(std::ostream& os, const TensorIndex& tensor_index);

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
  if (optional) {
    os << *optional;
  } else {
    os << "?";
  }
  return os;
}

template <typename T>
std::string ToString(
    const T* ptr, size_t n,
    size_t max_num_elem_to_print = std::numeric_limits<size_t>::max()) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < std::min(n, max_num_elem_to_print); ++i) {
    if (i > 0) {
      os << ", ";
    }
    if constexpr (std::is_same<T, int8_t>::value ||
                  std::is_same<T, uint8_t>::value) {
      os << "0x" << std::hex << static_cast<uint32_t>(ptr[i]);
    } else {
      os << ptr[i];
    }
  }
  if (n > max_num_elem_to_print) {
    os << ", ...";
  }
  os << "]";
  return os.str();
}

template <typename T, typename... Types>
inline std::string ToString(const std::vector<T, Types...>& vec) {
  return ToString(vec.data(), vec.size());
}

template <typename T, typename... Types>
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<T, Types...>& vec) {
  return os << ToString(vec.data(), vec.size());
}

template <typename T>
inline std::string ToString(absl::Span<T> span) {
  return ToString(span.data(), span.size());
}

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DEBUG_H_
