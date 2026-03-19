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

#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"

#include <sstream>
#include <string>
#include <type_traits>
#include <variant>

#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {

// Gets a string representation of the given element type.
std::string ToString(const QuantizedElementTypePerTensor& t) {
  std::stringstream sstr;
  sstr << "QuantizedPerTensor[" << ToString(t.StorageType()) << ", "
       << ToString(t.ExpressedType()) << "]";
  return sstr.str();
}

// Gets a string representation of the given element type.
std::string ToString(const QuantizedElementTypePerAxis& t) {
  std::stringstream sstr;
  sstr << "QuantizedPerAxis[" << ToString(t.StorageType()) << ", "
       << ToString(t.ExpressedType()) << ", " << t.QuantizedDimension() << "]";
  return sstr.str();
}

QuantizedElementTypePerTensor BaselineType(
    const QuantizedElementTypePerTensor& type) {
  QuantizedElementTypePerTensor baseline = type;
  std::visit(
      [](auto& scale) -> void {
        scale = std::remove_reference_t<decltype(scale)>(1);
      },
      baseline.Scale());
  std::visit(
      [](auto& zero_point) -> void {
        zero_point = std::remove_reference_t<decltype(zero_point)>(0);
      },
      baseline.ZeroPoint());
  return baseline;
}

QuantizedElementTypePerAxis BaselineType(
    const QuantizedElementTypePerAxis& type) {
  QuantizedElementTypePerAxis baseline = type;
  std::visit(
      [](auto& scales) -> void {
        using T = std::remove_reference_t<decltype(scales[0])>;
        absl::c_fill(scales, static_cast<T>(1));
      },
      baseline.Scales());
  std::visit(
      [](auto& zero_points) -> void {
        using T = std::remove_reference_t<decltype(zero_points[0])>;
        absl::c_fill(zero_points, static_cast<T>(0));
      },
      baseline.ZeroPoints());
  return baseline;
}

}  // namespace shlo_ref
