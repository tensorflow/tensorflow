/* Copyright 2026 The OpenXLA Authors.

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
#ifndef XLA_HLO_EVALUATOR_STACK_LITERAL_H_
#define XLA_HLO_EVALUATOR_STACK_LITERAL_H_

#include <cstdint>
#include <cstring>

#include "xla/xla_data.pb.h"

namespace xla {

// A lightweight, stack-allocated container to hold a chunk of data for
// vectorized interpretation in HloEvaluator.
// It holds up to 256 bytes of data, which is enough for 4 AVX-512 vectors.
class StackLiteral {
 public:
  StackLiteral() = default;

  explicit StackLiteral(PrimitiveType type) : type_(type) {
    std::memset(buffer_, 0, sizeof(buffer_));
  }

  PrimitiveType element_type() const { return type_; }

  void* untyped_data() { return buffer_; }
  const void* untyped_data() const { return buffer_; }

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(buffer_);
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(buffer_);
  }

  static constexpr size_t kBufferSize = 256;

 private:
  PrimitiveType type_ = PRIMITIVE_TYPE_INVALID;
  alignas(64) char buffer_[kBufferSize];
};

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_STACK_LITERAL_H_
