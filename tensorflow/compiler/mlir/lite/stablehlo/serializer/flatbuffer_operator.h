/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// prototype for stablehlo serialization, WIP
// WARNING: converting to stablehlo file is experimental feature, and no runtime
// support is provided

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_OPERATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_OPERATOR_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace odml {

// TODO(zichuanwei@): support float16/bfloat16 & int4

// Function calls with a non-specialized type will result to a linker error.
template <typename T>
inline std::vector<T> GetVector(DenseElementsAttr elements);

// TODO(zichuanwei@): for each type, we need to make sure the element type
// matches the expected type otherwise an error should be thrown, but for now
// we're just returning empty vector
template <>
inline std::vector<bool> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(1)) {
    auto vec = llvm::to_vector(
        llvm::map_range(elements.getValues<bool>(),
                        [&](bool value) -> uint8_t { return value ? 1 : 0; }));
    return std::vector<bool>(vec.begin(), vec.end());
  }

  return std::vector<bool>();
}

template <>
inline std::vector<int8_t> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(8)) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APInt>(),
        [&](APInt value) -> int8_t { return value.getSExtValue(); }));
    return std::vector<int8_t>(vec.begin(), vec.end());
  }

  return std::vector<int8_t>();
}

template <>
inline std::vector<int16_t> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(16)) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APInt>(),
        [&](APInt value) -> int16_t { return value.getSExtValue(); }));
    return std::vector<int16_t>(vec.begin(), vec.end());
  }

  return std::vector<int16_t>();
}

template <>
inline std::vector<int32_t> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(32)) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APInt>(),
        [&](APInt value) -> int32_t { return value.getSExtValue(); }));
    return std::vector<int32_t>(vec.begin(), vec.end());
  }

  return std::vector<int32_t>();
}

template <>
inline std::vector<int64_t> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(64)) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APInt>(),
        [&](APInt value) -> int64_t { return value.getSExtValue(); }));
    return std::vector<int64_t>(vec.begin(), vec.end());
  }

  return std::vector<int64_t>();
}

template <>
inline std::vector<float> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isF32()) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APFloat>(),
        [&](APFloat value) -> float { return value.convertToFloat(); }));
    return std::vector<float>(vec.begin(), vec.end());
  }

  return std::vector<float>();
}

template <>
inline std::vector<double> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isF64()) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APFloat>(),
        [&](APFloat value) -> double { return value.convertToFloat(); }));
    return std::vector<double>(vec.begin(), vec.end());
  }

  return std::vector<double>();
}

// Handles the case when the DenseElementsAttr doesn't exist, and when it
// doesn't returns a vector of length `default_size` all with the same value
// `default_value`.
template <typename T>
static inline std::vector<T> GetOptionalVector(
    std::optional<DenseElementsAttr> elements, int64_t default_size,
    int64_t default_value) {
  if (elements.has_value()) {
    return GetVector<T>(elements.value());
  }
  return std::vector<T>(default_size, default_value);
}

}  // namespace odml
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_OPERATOR_H_
