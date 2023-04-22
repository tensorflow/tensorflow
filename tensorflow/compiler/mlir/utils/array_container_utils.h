/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_

#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

template <typename T>
inline llvm::ArrayRef<T> SpanToArrayRef(absl::Span<const T> span) {
  return llvm::ArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline llvm::ArrayRef<T> SpanToArrayRef(absl::Span<T> span) {
  return llvm::ArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline llvm::MutableArrayRef<T> SpanToMutableArrayRef(absl::Span<T> span) {
  return llvm::MutableArrayRef<T>(span.data(), span.size());
}

template <typename T>
inline absl::Span<const T> ArrayRefToSpan(llvm::ArrayRef<T> ref) {
  return absl::Span<const T>(ref.data(), ref.size());
}

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_UTILS_ARRAY_CONTAINER_UTILS_H_
