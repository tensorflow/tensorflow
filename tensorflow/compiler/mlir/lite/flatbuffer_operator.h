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
#include "absl/status/status.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloTypes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {

// duplicated from
// https://github.com/openxla/stablehlo/blob/e5ad51715a11721c78b6748ab5de7945df24b1b8/stablehlo/transforms/StablehloLegalizeToVhlo.cpp#L756
// so we can create correct vhlo types
class StablehloVhloTypeConverter : public mlir::vhlo::VhloTypeConverter {
 public:
  StablehloVhloTypeConverter() : mlir::vhlo::VhloTypeConverter() {
    addConversion([](mlir::Type type) -> mlir::Type {
      if (type.getDialect().getNamespace() ==
          mlir::vhlo::VhloDialect::getDialectNamespace()) {
        return type;
      }
      return {};
    });
    addConversion([](mlir::stablehlo::TokenType token) -> mlir::Type {
      return mlir::vhlo::TokenV1Type::get(token.getContext());
    });
    addBuiltinToVhloConversions();
  }

  mlir::Attribute convertEncoding(mlir::Attribute attr) const final {
    // Must be VHLO encoding, or convertible to VHLO encoding.
    if (attr.getDialect().getNamespace() ==
        mlir::vhlo::VhloDialect::getDialectNamespace())
      return attr;

    if (auto stablehloAttr =
            mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(attr)) {
      return mlir::vhlo::TypeExtensionsV1Attr::get(stablehloAttr.getContext(),
                                                   stablehloAttr.getBounds());
    }

    // Was not VHLO encoding, or convertible.
    return {};
  }
};

// from
// https://github.com/openxla/stablehlo/blob/e5ad51715a11721c78b6748ab5de7945df24b1b8/stablehlo/transforms/VhloLegalizeToStablehlo.cpp#L45C70-L45C70
class VhloToStablehloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  VhloToStablehloTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenV1Type token) -> Type {
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();
  }

  Attribute convertEncoding(Attribute attr) const final {
    if (auto vhloAttr =
            mlir::dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>(attr)) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    // All encodings supported in StableHLO.
    return attr;
  }
};

// Returns true if the op_code belongs to a stablehlo operation.
bool IsStablehloOp(const tflite::OperatorCodeT &op_code);

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
    flatbuffers::FlatBufferBuilder *fbb,
    std::optional<int> debug_metadata_index = -1);

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
absl::Status CustomOptionsToAttributes(
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

// Handles the case when the ArrayRef doesn't exist, and when it
// doesn't returns a vector of length `default_size` all with the same value
// `default_value`.
template <typename T>
static inline std::vector<T> GetOptionalVector(
    std::optional<ArrayRef<T>> values, int64_t default_size = 0,
    int64_t default_value = 0) {
  if (values.has_value()) {
    return std::vector<T>(values->begin(), values->end());
  }
  return std::vector<T>(default_size, default_value);
}

template <typename T>
static inline std::vector<T> GetVector(
    vhlo::TensorV1Attr elements,
    mlir::vhlo::VhloTypeConverter &vhlo_type_converter) {
  return GetOptionalVector<T>(mlir::DenseIntElementsAttr::getFromRawBuffer(
      mlir::cast<mlir::ShapedType>(
          vhlo_type_converter.convertType(elements.getType())),
      elements.getData()));
}

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
