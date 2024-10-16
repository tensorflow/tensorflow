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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_MODEL_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

namespace lrt {

// Data type of tensor elements. C++ equivalent to LrtElementType.
enum class ElementType {
  None = kLrtElementTypeNone,
  Bool = kLrtElementTypeBool,
  Int4 = kLrtElementTypeInt4,
  Int8 = kLrtElementTypeInt8,
  Int16 = kLrtElementTypeInt16,
  Int32 = kLrtElementTypeInt32,
  Int64 = kLrtElementTypeInt64,
  UInt8 = kLrtElementTypeUInt8,
  UInt16 = kLrtElementTypeUInt16,
  UInt32 = kLrtElementTypeUInt32,
  UInt64 = kLrtElementTypeUInt64,
  Float16 = kLrtElementTypeFloat16,
  BFloat16 = kLrtElementTypeBFloat16,
  Float32 = kLrtElementTypeFloat32,
  Float64 = kLrtElementTypeFloat64,
  Complex64 = kLrtElementTypeComplex64,
  Complex128 = kLrtElementTypeComplex128,
  TfResource = kLrtElementTypeTfResource,
  TfString = kLrtElementTypeTfString,
  TfVariant = kLrtElementTypeTfVariant,
};

// Tensor layout. C++ equivalent to LrtLayout.
class Layout {
 public:
  explicit Layout(std::vector<int32_t>&& dimensions,
                  std::vector<uint32_t>&& strides = std::vector<uint32_t>())
      : dimensions_(std::move(dimensions)), strides_(std::move(strides)) {}

  explicit Layout(const LrtLayout& layout)
      : dimensions_(layout.dimensions, layout.dimensions + layout.rank) {
    if (layout.strides) {
      strides_.reserve(layout.rank);
      std::copy(layout.strides, layout.strides + layout.rank,
                std::back_inserter(strides_));
    }
  }

  explicit operator LrtLayout() const {
    return LrtLayout{
        /*.rank=*/Rank(),
        /*.dimensions=*/dimensions_.data(),
        /*.strides=*/(HasStrides() ? strides_.data() : nullptr),
    };
  }

  bool operator==(const Layout& other) const {
    return dimensions_ == other.dimensions_ && strides_ == other.strides_;
  }

  uint32_t Rank() const { return dimensions_.size(); }

  absl::Span<const int32_t> Dimensions() const {
    return absl::MakeSpan(dimensions_.data(), dimensions_.size());
  }

  bool HasStrides() const { return !strides_.empty(); }

  absl::Span<const uint32_t> Strides() const {
    const uint32_t* data = HasStrides() ? strides_.data() : nullptr;
    auto size = HasStrides() ? Rank() : 0;
    return absl::MakeSpan(data, size);
  }

 private:
  std::vector<int32_t> dimensions_;
  std::vector<uint32_t> strides_;
};

// Type for tensors with known dimensions. C++ equivalent to
// LrtRankedTensorType.
class RankedTensorType {
 public:
  RankedTensorType(ElementType element_type, Layout&& layout)
      : element_type_(element_type), layout_(std::move(layout)) {}
  explicit RankedTensorType(const LrtRankedTensorType& type)
      : element_type_(static_cast<enum ElementType>(type.element_type)),
        layout_(type.layout) {}

  explicit operator LrtRankedTensorType() const {
    return LrtRankedTensorType{
        /*.element_type=*/static_cast<LrtElementType>(element_type_),
        /*layout=*/static_cast<LrtLayout>(layout_),
    };
  }

  bool operator==(const RankedTensorType& other) const {
    return ElementType() == other.ElementType() && Layout() == other.Layout();
  }

  ElementType ElementType() const { return element_type_; }

  const Layout& Layout() const { return layout_; }

 private:
  enum ElementType element_type_;
  class Layout layout_;
};

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_MODEL_H_
