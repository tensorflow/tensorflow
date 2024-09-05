/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ATTRIBUTE_SPAN_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ATTRIBUTE_SPAN_H_

#include <cstring>
#include <type_traits>

#include "absl/log/check.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/iterator.h"

namespace mlrt {
namespace attribute_internal {

// LINT.IfChange(mlrt_attributes)
template <typename T>
inline constexpr bool kCanAttributeBeInlined =
    (std::is_integral_v<T> ||
     std::is_floating_point_v<T>)&&(sizeof(T) <= sizeof(uint32_t));
// LINT.ThenChange(../../../../compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.cc:mlrt_attributes)

}  // namespace attribute_internal

class AttributeSpan {
  class Iterator
      : public iterator_internal::IteratorBase<Iterator, bc::String,
                                               bc::Span<bc::String>> {
   public:
    using IteratorBase<Iterator, bc::String,
                       bc::Span<bc::String>>::IteratorBase;
  };

 public:
  using value_type = bc::String;
  using iterator = Iterator;
  using const_iterator = iterator;

  AttributeSpan(bc::Span<uint32_t> attr_indices,
                bc::Span<bc::String> attributes)
      : attr_indices_(attr_indices), attributes_(attributes) {}

  bc::String operator[](size_t id) const {
    return attributes_[attr_indices_[id]];
  }

  template <typename T>
  T GetAs(size_t id) const {
    if constexpr (std::is_same_v<T, bc::String>) {
      return attributes_[attr_indices_[id]];
    }

    if constexpr (attribute_internal::kCanAttributeBeInlined<T>) {
      return bc::AccessTraits<T>::Read(attr_indices_.data(id));
    }

    return bc::AccessTraits<T>::Read(attributes_[attr_indices_[id]].data());
  }

  size_t size() const { return attr_indices_.size(); }

  iterator begin() const {
    return iterator(attr_indices_.begin(), attributes_);
  }
  iterator end() const { return iterator(attr_indices_.end(), attributes_); }

 private:
  bc::Span<uint32_t> attr_indices_;
  bc::Span<bc::String> attributes_;
};

}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_ATTRIBUTE_SPAN_H_
