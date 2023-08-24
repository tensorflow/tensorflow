/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

namespace mlir {

// Returns the MLIR op name for the flatbuffer operator corresponding to
// `op_code`.
std::string GetMlirOpNameFromOpCode(const ::tflite::OperatorCodeT &op_code);

// Returns the builtin op code for the given MLIR operation on success; emits
// error and returns std::nullopt on failure.
std::optional<tflite::BuiltinOperator> GetBuiltinOpCode(Operation *mlir_op);

// Packs the given MLIR operation into a TFLite FlatBuffer operator object.
// Returns the FlatBuffer offset for the operator on success; emits error and
// returns std::nullopt on failure.
std::optional<flatbuffers::Offset<tflite::Operator>> CreateFlatBufferOperator(
    Operation *mlir_op, uint32_t opcode_index,
    const std::vector<int32_t> &operands, const std::vector<int32_t> &results,
    const std::vector<int32_t> &intermediates,
    flatbuffers::FlatBufferBuilder *fbb);

// Populates the array of mlir::NamedAttributes corresponding to the given
// tflite::FlatbufferOptionsUnion.
// We use an out parameter per LLVM convention
void BuiltinOptionsToAttributes(
    tflite::BuiltinOptionsUnion op_union, mlir::Builder builder,
    // NOLINTNEXTLINE
    llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);

// While the last several tensors could be optional tensors for an tfl op, the
// number of input operands could vary. This function gets the min/max number of
// operands from tflite op name.
llvm::MinMax OperandNumbersMinMax(llvm::StringRef op_name);

// Populates the `custom_code` and `custom_options` to attributes.
// `custom_code` is used to identify CustomOp.
// `custom_options` are opaque attribute used to store infomations for this
// custom op.
tensorflow::Status CustomOptionsToAttributes(
    const std::string &custom_code, const std::vector<uint8_t> &custom_options,
    mlir::Builder builder,
    // NOLINTNEXTLINE
    Location loc, llvm::SmallVectorImpl<mlir::NamedAttribute> *attributes);

// TODO(zichuanwei@): Populate Builtin_options_2 manual for now, should automate
// these in the future
void BuiltinOptions2ToAttributes(
    tflite::BuiltinOptions2Union op_union, mlir::Builder builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);

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
inline std::vector<uint64_t> GetVector(DenseElementsAttr elements) {
  auto type = elements.getType();
  auto elemType = type.getElementType();
  if (elemType.isSignlessInteger(64)) {
    auto vec = llvm::to_vector(llvm::map_range(
        elements.getValues<APInt>(),
        [&](APInt value) -> uint64_t { return value.getSExtValue(); }));
    return std::vector<uint64_t>(vec.begin(), vec.end());
  }

  return std::vector<uint64_t>();
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
    std::optional<DenseElementsAttr> elements, int64_t default_size = 0,
    int64_t default_value = 0) {
  if (elements.has_value()) {
    return GetVector<T>(elements.value());
  }
  return std::vector<T>(default_size, default_value);
}

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
