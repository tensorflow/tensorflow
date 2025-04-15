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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"

namespace stablehlo {

template <ElementType storage_type, ElementType expressed_type>
inline typename Storage<expressed_type>::Type Dequantize(
    typename Storage<storage_type>::Type quantized_value,
    const QuantizedParameter& quant_param) {
  using ST = typename Storage<storage_type>::Type;
  using ET = typename Storage<expressed_type>::Type;
  auto sub = (quantized_value - static_cast<ST>(quant_param.zero_point));
  return static_cast<ET>(sub) * static_cast<ET>(quant_param.scale);
}

template <ElementType storage_type, ElementType expressed_type>
inline typename Storage<storage_type>::Type QuantizePartial(
    typename Storage<expressed_type>::Type expressed_value,
    typename Storage<expressed_type>::Type scale_inv,
    typename Storage<storage_type>::Type zero_point) {
  using ST = typename Storage<storage_type>::Type;
  using ET = typename Storage<expressed_type>::Type;
  ET rounding_extra = (expressed_value >= 0) ? ET(0.5) : ET(-0.5);
  ET tmp = (expressed_value * scale_inv + rounding_extra);
  // Clamp the value in case of overflow/underflow. This is needed to avoid
  // getting a SIGILL exception when casting down below.
  ET max = std::numeric_limits<ST>::max();
  ET min = std::numeric_limits<ST>::min();
  if (tmp > max) {
    tmp = max;
  } else if (tmp < min) {
    tmp = min;
  }
  ST rounded_value = static_cast<ST>(tmp);
  ST storage_value = rounded_value + zero_point;
  return storage_value;
}

template <ElementType storage_type>
absl::Status CompleteQuantization(void* buffer, size_t n,
                                  const std::optional<int32_t>& storage_min,
                                  const std::optional<int32_t>& storage_max) {
  using S = Storage<storage_type>;

  if (storage_min) {
    typename S::Type min = *storage_min;
    for (size_t i = 0; i < n; ++i) {
      auto storage = S::Get(buffer, i);
      storage = std::max(storage, min);
      S::Set(buffer, i, storage);
    }
  }

  if (storage_max) {
    typename S::Type max = *storage_max;
    for (size_t i = 0; i < n; ++i) {
      auto storage = S::Get(buffer, i);
      storage = std::min(storage, max);
      S::Set(buffer, i, storage);
    }
  }

  return absl::OkStatus();
}

template <ElementType storage_type>
absl::Status CompleteQuantization(QuantizedTensor& result) {
  if (storage_type != result.storage_type()) {
    return absl::InvalidArgumentError("Unexpected storage type");
  }

  size_t n = result.num_elements();
  auto result_buffer = result.buffer();
  const auto& result_storage_min = result.type().element_type().storage_min();
  const auto& result_storage_max = result.type().element_type().storage_max();
  return CompleteQuantization<storage_type>(
      result_buffer, n, result_storage_min, result_storage_max);
}

// /////////////////////////////////////////////////////////////////////////////

template <ElementType storage_type, ElementType expressed_type, typename Op>
inline typename Storage<storage_type>::Type DequantizeOpQuantizePartial(
    typename Storage<expressed_type>::Type operand_storage,
    const QuantizedParameter& operand_quant_param,
    typename Storage<expressed_type>::Type result_scale_inv,
    typename Storage<storage_type>::Type result_zero_point, Op&& op) {
  auto operand_expressed = Dequantize<storage_type, expressed_type>(
      operand_storage, operand_quant_param);
  auto result_expressed = op(operand_expressed);
  return QuantizePartial<storage_type, expressed_type>(
      result_expressed, result_scale_inv, result_zero_point);
}

template <ElementType storage_type, ElementType expressed_type, typename Op>
inline typename Storage<storage_type>::Type DequantizeOpQuantizePartial(
    typename Storage<expressed_type>::Type lhs_storage,
    typename Storage<expressed_type>::Type rhs_storage,
    const QuantizedParameter& lhs_quant_param,
    const QuantizedParameter& rhs_quant_param,
    typename Storage<expressed_type>::Type result_scale_inv,
    typename Storage<storage_type>::Type result_zero_point, Op&& op) {
  auto lhs_expressed =
      Dequantize<storage_type, expressed_type>(lhs_storage, lhs_quant_param);
  auto rhs_expressed =
      Dequantize<storage_type, expressed_type>(rhs_storage, rhs_quant_param);
  auto result_expressed = op(lhs_expressed, rhs_expressed);
  return QuantizePartial<storage_type, expressed_type>(
      result_expressed, result_scale_inv, result_zero_point);
}

// /////////////////////////////////////////////////////////////////////////////

class TensorIndex {
 public:
  explicit TensorIndex(const Shape& shape)
      : shape_(shape), index_(shape.rank(), 0), linear_index_(0) {}
  auto operator[](size_t idx) const { return index_[idx]; }
  void set(size_t idx, DimensionSize value) {
    index_[idx] = value;
    linear_index_.reset();
  }
  void set(const TensorIndex& other) {
    index_ = other.index_;
    linear_index_.reset();
  }
  // Return a linearized index assuming the shape's dimension 0 is the major
  // index (i.e., slowest moving dimension) and the shape's dimension R-1, where
  // R is the rank, is the minor index (i.e., the fastest moving dimension). For
  // instance, for tensor<2x3xf32> dimension 1 (of size 3) is the fastest moving
  // dimension.
  size_t linearize() const {
    if (!linear_index_) {
      linear_index_ = compute_linear_index();
    }
    return *linear_index_;
  }

 private:
  friend class TensorIndexIterator;
  friend std::ostream& operator<<(std::ostream&, const TensorIndex&);

  size_t compute_linear_index() const {
    auto n = index_.size();
    size_t linear_index = 0;
    for (auto i = 0; i < n; ++i) {
      linear_index = (linear_index * shape_.dim(i)) + index_[i];
    }
    return linear_index;
  }

  bool advance() {
    auto n = index_.size();
    index_[n - 1]++;
    if (linear_index_) {
      (*linear_index_)++;
    }
    for (int i = n - 1; i >= 0; --i) {
      if (index_[i] == shape_.dim(i)) {
        if ((i - 1) >= 0) {
          index_[i] = 0;
          index_[i - 1]++;
        } else {
          // Overflow.
          return false;
        }
      }
    }
    return true;
  }

  Shape shape_;
  std::vector<DimensionSize> index_;
  mutable std::optional<size_t> linear_index_;
};

class TensorIndexIterator {
 public:
  explicit TensorIndexIterator(const Shape& shape) : index_(shape) {}
  TensorIndexIterator& operator++() {
    has_next_ = index_.advance();
    return *this;
  }
  const TensorIndex& operator*() const { return index_; }
  const TensorIndex* operator->() const { return &index_; }
  bool has_next() const { return has_next_; }

 private:
  TensorIndex index_;
  bool has_next_ = true;
};

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_UTIL_H_
