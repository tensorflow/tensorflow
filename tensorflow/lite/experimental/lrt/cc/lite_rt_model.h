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
#include <utility>

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
  Ing64 = kLrtElementTypeInt64,
  UInt8 = kLrtElementTypeUInt8,
  UInt16 = kLrtElementTypeUInt16,
  UInt32 = kLrtElementTypeUInt32,
  UInt64 = kLrtElementTypeUInt64,
  Float16 = kLrtElementTypeFloat16,
  BFloat16 = kLrtElementTypeBFloat16,
  Float32 = kLrtElementTypeFloat32,
  BFloat64 = kLrtElementTypeFloat64,
  Complex64 = kLrtElementTypeComplex64,
  Complex1128 = kLrtElementTypeComplex128,
  TfResource = kLrtElementTypeTfResource,
  TfString = kLrtElementTypeTfString,
  TfVariant = kLrtElementTypeTfVariant,
};

// Tensor layout. C++ equivalent to LrtLayout.
class Layout {
 public:
  explicit Layout(const LrtLayout& layout) : layout_(layout) {}
  explicit Layout(LrtLayout&& layout) : layout_(std::move(layout)) {}

  explicit operator const LrtLayout&() const { return layout_; }

  bool operator==(const Layout& other) const {
    return Rank() == other.Rank() &&
           std::equal(Dimensions().begin(), Dimensions().end(),
                      other.Dimensions().begin()) &&
           (HasStrides() == other.HasStrides()) &&
           std::equal(Strides().begin(), Strides().end(),
                      other.Strides().begin());
  }

  uint32_t Rank() const { return layout_.rank; }

  absl::Span<const int32_t> Dimensions() const {
    return absl::MakeSpan(layout_.dimensions, layout_.rank);
  }

  bool HasStrides() const { return layout_.strides != nullptr; }

  absl::Span<const uint32_t> Strides() const {
    auto num_strides = HasStrides() ? Rank() : 0;
    return absl::MakeSpan(layout_.strides, num_strides);
  }

 private:
  LrtLayout layout_;
};

// Type for tensors with known dimensions. C++ equivalent to
// LrtRankedTensorType.
class RankedTensorType {
 public:
  explicit RankedTensorType(const LrtRankedTensorType& type) : type_(type) {}
  explicit RankedTensorType(LrtRankedTensorType&& type)
      : type_(std::move(type)) {}

  explicit operator const LrtRankedTensorType&() const { return type_; }

  bool operator==(const RankedTensorType& other) const {
    return ElementType() == other.ElementType() && Layout() == other.Layout();
  }

  ElementType ElementType() const {
    return static_cast<enum ElementType>(type_.element_type);
  }

  Layout Layout() const { return lrt::Layout(type_.layout); }

 private:
  LrtRankedTensorType type_;
};

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_MODEL_H_
