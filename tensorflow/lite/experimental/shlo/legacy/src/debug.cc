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

#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/bf16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/f16.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/util.h"

namespace stablehlo {

namespace {

template <typename Float>
inline constexpr Float MaxRelativeDifference() {
  if constexpr (std::is_same_v<Float, BF16>) {
    return 1e-2;
  } else if constexpr (std::is_same_v<Float, F16>) {
    return 2e-3;
  } else {
    return 1e-6;
  }
}

// Comare two floats by computing the relative difference. See
// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
template <typename Float>
bool AlmostSame(
    Float input1, Float input2,
    Float max_relative_difference = MaxRelativeDifference<Float>()) {
  bool result;
  if (std::isnan(input1) || std::isnan(input2)) {
    result = (std::isnan(input1) and std::isnan(input2));
  } else if (std::isinf(input1) || std::isinf(input2)) {
    result = (std::isinf(input1) and std::isinf(input2)) and
             (std::signbit(static_cast<float>(input1)) ==
              std::signbit(static_cast<float>(input2)));
  } else if (input1 == Float(0.0f)) {
    result = input2 < max_relative_difference;
  } else if (input2 == Float(0.0f)) {
    result = input1 < max_relative_difference;
  } else {
    Float diff = input1 - input2;
    if (diff < 0) {
      diff = -diff;
    }
    Float a = (input1 > 0) ? input1 : -input1;
    Float b = (input2 > 0) ? input2 : -input2;
    Float largest = (a > b) ? a : b;
    Float max_diff = (largest * max_relative_difference);
    result = diff < max_diff;
  }
  return result;
}

template <typename Float>
bool AlmostSame(
    const Float* input1, const Float* input2, size_t num_elements,
    Float max_relative_difference = MaxRelativeDifference<Float>()) {
  for (size_t i = 0; i < num_elements; ++i) {
    if (!AlmostSame<Float>(input1[i], input2[i], max_relative_difference)) {
      return false;
    }
  }
  return true;
}

}  // namespace

// /////////////////////////////////////////////////////////////////////////////

bool AlmostSame(const Tensor& x, const Tensor& y) {
  if (x.type() != y.type()) {
    return false;
  }

  switch (x.element_type()) {
    case ElementType::kI1:
    case ElementType::kSI8:
    case ElementType::kSI16:
    case ElementType::kSI32:
      return !std::memcmp(x.buffer(), y.buffer(), x.num_bytes());
    case ElementType::kBF16: {
      using ET = typename Storage<ElementType::kBF16>::Type;
      return AlmostSame(static_cast<const ET*>(x.buffer()),
                        static_cast<const ET*>(y.buffer()), x.num_elements());
    }
    case ElementType::kF16: {
      using ET = typename Storage<ElementType::kF16>::Type;
      return AlmostSame(static_cast<const ET*>(x.buffer()),
                        static_cast<const ET*>(y.buffer()), x.num_elements());
    }
    case ElementType::kF32: {
      using ET = typename Storage<ElementType::kF32>::Type;
      return AlmostSame(static_cast<const ET*>(x.buffer()),
                        static_cast<const ET*>(y.buffer()), x.num_elements());
    }
    default:
      LOG(ERROR) << "Unexpected tensor element type" << x.element_type();
      return false;
  }
}

bool AlmostSame(const QuantizedTensor& x, const QuantizedTensor& y) {
  if (x.type() != y.type()) {
    return false;
  }

  // For now we support only per-tensor quantization.
  CHECK(x.is_per_tensor_quantized());  // Crash OK

  const QuantizedParameter& quant_param = x.type().element_type().parameters(0);

  auto x_buffer = x.buffer();
  auto y_buffer = y.buffer();

  size_t n = x.num_elements();
  for (size_t i = 0; i < n; ++i) {
    switch (x.storage_type()) {
      case ElementType::kSI8: {
        auto x_quantized_value = Storage<ElementType::kSI8>::Get(x_buffer, i);
        auto y_quantized_value = Storage<ElementType::kSI8>::Get(y_buffer, i);
        switch (x.expressed_type()) {
          case ElementType::kBF16: {
            auto x_value = Dequantize<ElementType::kSI8, ElementType::kBF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI8, ElementType::kBF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF16: {
            auto x_value = Dequantize<ElementType::kSI8, ElementType::kF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI8, ElementType::kF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF32: {
            auto x_value = Dequantize<ElementType::kSI8, ElementType::kF32>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI8, ElementType::kF32>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << x.expressed_type();
            return false;
        }
      } break;

      case ElementType::kSI16: {
        auto x_quantized_value = Storage<ElementType::kSI16>::Get(x_buffer, i);
        auto y_quantized_value = Storage<ElementType::kSI16>::Get(y_buffer, i);
        switch (x.expressed_type()) {
          case ElementType::kBF16: {
            auto x_value = Dequantize<ElementType::kSI16, ElementType::kBF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI16, ElementType::kBF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF16: {
            auto x_value = Dequantize<ElementType::kSI16, ElementType::kF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI16, ElementType::kF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF32: {
            auto x_value = Dequantize<ElementType::kSI16, ElementType::kF32>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI16, ElementType::kF32>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << x.expressed_type();
            return false;
        }
      } break;

      case ElementType::kSI32: {
        auto x_quantized_value = Storage<ElementType::kSI32>::Get(x_buffer, i);
        auto y_quantized_value = Storage<ElementType::kSI32>::Get(y_buffer, i);
        switch (x.expressed_type()) {
          case ElementType::kBF16: {
            auto x_value = Dequantize<ElementType::kSI32, ElementType::kBF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI32, ElementType::kBF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF16: {
            auto x_value = Dequantize<ElementType::kSI32, ElementType::kF16>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI32, ElementType::kF16>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          case ElementType::kF32: {
            auto x_value = Dequantize<ElementType::kSI32, ElementType::kF32>(
                x_quantized_value, quant_param);
            auto y_value = Dequantize<ElementType::kSI32, ElementType::kF32>(
                y_quantized_value, quant_param);
            if (!AlmostSame(x_value, y_value)) {
              return false;
            }
          } break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << x.expressed_type();
            return false;
        }
      } break;

      default:
        LOG(ERROR) << "Unexpected storage type: " << x.storage_type();
        return false;
    }
  }

  return true;
}

// /////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, ElementType element_type) {
  switch (element_type) {
    case ElementType::kI1:
      return os << "i1";
    case ElementType::kSI8:
      return os << "si8";
    case ElementType::kSI16:
      return os << "si16";
    case ElementType::kSI32:
      return os << "si32";
    case ElementType::kBF16:
      return os << "bf16";
    case ElementType::kF16:
      return os << "f16";
    case ElementType::kF32:
      return os << "f32";
    default:
      LOG(ERROR) << "Unexpected element type: "
                 << static_cast<int>(element_type);
      return os;
  }
}

std::ostream& operator<<(std::ostream& os, const Shape& s) {
  for (auto i = 0; i < s.rank(); ++i) {
    os << s.dim(i) << "x";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorType& t) {
  return os << "tensor<" << t.shape() << t.element_type() << ">";
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << t.type() << ": ";

  size_t n = t.num_elements();
  auto t_buffer = t.buffer();

  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    switch (t.element_type()) {
      case ElementType::kI1:
        os << static_cast<bool>(Storage<ElementType::kI1>::Get(t_buffer, i));
        break;
      case ElementType::kSI8:
        os << static_cast<int>(Storage<ElementType::kSI8>::Get(t_buffer, i));
        break;
      case ElementType::kSI16:
        os << Storage<ElementType::kSI16>::Get(t_buffer, i);
        break;
      case ElementType::kSI32:
        os << Storage<ElementType::kSI32>::Get(t_buffer, i);
        break;
      case ElementType::kBF16:
        os << Storage<ElementType::kBF16>::Get(t_buffer, i);
        break;
      case ElementType::kF16:
        os << Storage<ElementType::kF16>::Get(t_buffer, i);
        break;
      case ElementType::kF32:
        os << Storage<ElementType::kF32>::Get(t_buffer, i);
        break;
      default:
        LOG(ERROR) << "Unexpected element type: " << t.element_type();
        break;
    }
  }

  return os;
}

// /////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os,
                         const QuantizedTensorElementType& t) {
  os << "!quant.uniform<" << t.storage_type();
  os << ":" << t.expressed_type();
  if (t.storage_min() || t.storage_max()) {
    int32_t min;
    int32_t max;
    switch (t.storage_type()) {
      case ElementType::kSI8:
        min = std::numeric_limits<int8_t>::min();
        max = std::numeric_limits<int8_t>::max();
        break;
      case ElementType::kSI16:
        min = std::numeric_limits<int16_t>::min();
        max = std::numeric_limits<int16_t>::max();
        break;
      case ElementType::kSI32:
        min = std::numeric_limits<int32_t>::min();
        max = std::numeric_limits<int32_t>::max();
        break;
      default:
        LOG(ERROR) << "Unexpected storage type: " << t.storage_type();
        min = std::numeric_limits<int32_t>::min();
        max = std::numeric_limits<int32_t>::max();
        break;
    }
    os << "<" << t.storage_min().value_or(min) << ":"
       << t.storage_max().value_or(max) << ">";
  }
  if (t.quantized_dimension()) {
    os << ":" << *t.quantized_dimension();
  }
  os << ", ";
  if (t.is_per_tensor_quantized()) {
    const auto& p = t.parameters(0);
    os << p.scale << ":" << p.zero_point;
  } else {
    os << "{";
    for (size_t i = 0; i < t.num_parameters(); ++i) {
      if (i > 0) {
        os << ",";
      }
      const auto& p = t.parameters(i);
      os << p.scale << ":" << p.zero_point;
    }
    os << "}";
  }
  return os << ">";
}

std::ostream& operator<<(std::ostream& os, const QuantizedTensorType& t) {
  return os << "tensor<" << t.shape() << t.element_type() << ">";
}

std::ostream& operator<<(std::ostream& os, const QuantizedTensor& t) {
  os << t.type() << ": ";

  if (t.is_per_axis_quantized()) {
    return os << "<per axis quantization>";
  }

  const QuantizedParameter& quant_param = t.type().element_type().parameters(0);
  auto t_buffer = t.buffer();

  size_t n = t.num_elements();
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }

    switch (t.storage_type()) {
      case ElementType::kSI8: {
        auto quantized_value = Storage<ElementType::kSI8>::Get(t_buffer, i);
        os << static_cast<int>(quantized_value) << "(";
        switch (t.expressed_type()) {
          case ElementType::kBF16:
            os << Dequantize<ElementType::kSI8, ElementType::kBF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF16:
            os << Dequantize<ElementType::kSI8, ElementType::kF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF32:
            os << Dequantize<ElementType::kSI8, ElementType::kF32>(
                quantized_value, quant_param);
            break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << t.expressed_type();
        }
        os << ")";
      } break;

      case ElementType::kSI16: {
        auto quantized_value = Storage<ElementType::kSI16>::Get(t_buffer, i);
        os << quantized_value << "(";
        switch (t.expressed_type()) {
          case ElementType::kBF16:
            os << Dequantize<ElementType::kSI16, ElementType::kBF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF16:
            os << Dequantize<ElementType::kSI16, ElementType::kF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF32:
            os << Dequantize<ElementType::kSI16, ElementType::kF32>(
                quantized_value, quant_param);
            break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << t.expressed_type();
        }
        os << ")";
      } break;

      case ElementType::kSI32: {
        auto quantized_value = Storage<ElementType::kSI32>::Get(t_buffer, i);
        os << quantized_value << "(";
        switch (t.expressed_type()) {
          case ElementType::kBF16:
            os << Dequantize<ElementType::kSI32, ElementType::kBF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF16:
            os << Dequantize<ElementType::kSI32, ElementType::kF16>(
                quantized_value, quant_param);
            break;
          case ElementType::kF32:
            os << Dequantize<ElementType::kSI32, ElementType::kF32>(
                quantized_value, quant_param);
            break;
          default:
            LOG(ERROR) << "Unexpected expressed type: " << t.expressed_type();
        }
        os << ")";
      } break;

      default:
        LOG(ERROR) << "Unexpected storage type: " << t.storage_type();
        break;
    }
  }

  return os;
}

// /////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os,
                         ComparisonDirection comparison_direction) {
  switch (comparison_direction) {
    case ComparisonDirection::kEQ:
      return os << "=";
    case ComparisonDirection::kNE:
      return os << "!=";
    case ComparisonDirection::kGE:
      return os << ">=";
    case ComparisonDirection::kGT:
      return os << ">";
    case ComparisonDirection::kLE:
      return os << "<=";
    case ComparisonDirection::kLT:
      return os << "<";
  }
}

std::ostream& operator<<(std::ostream& os, CompareType compare_type) {
  switch (compare_type) {
    case CompareType::kSigned:
      return os << "signed";
    case CompareType::kUnsigned:
      return os << "unsigned";
    case CompareType::kFloat:
      return os << "float";
    case CompareType::kTotalOrder:
      return os << "totalorder";
  }
}

// /////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, const TensorIndex& tensor_index) {
  return os << "[" << ToString(tensor_index.index_) << "->"
            << tensor_index.linear_index_ << "]";
}

}  // namespace stablehlo
