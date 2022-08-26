/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/utils/to_string.h"

namespace xla {
namespace runtime {

using absl::InvalidArgumentError;
using absl::StatusOr;
using absl::StrFormat;

// Type conversion for the canonical MLIR types supported by the runtime.
static std::unique_ptr<Type> ConvertCanonicalType(
    mlir::Type type, const TypeConverter& convert) {
  // KernelContextType -> KernelContextOperandType (both in xla::runtime).
  if (auto ctx = type.dyn_cast<KernelContextType>())
    return std::make_unique<KernelContextOperandType>();

  // mlir::async::TokenType -> xla::runtime::AsyncTokenType
  if (type.isa<mlir::async::TokenType>())
    return std::make_unique<AsyncTokenType>();

  // mlir::async::ValueType -> xla::runtime::AsyncValueType
  if (auto value = type.dyn_cast<mlir::async::ValueType>()) {
    if (auto value_type = convert.Convert(value.getValueType());
        value_type.ok())
      return std::make_unique<AsyncValueType>(std::move(*value_type));
  }

  // mlir::RankedTensorType -> xla::runtime::RankedTensorType
  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType());
        dtype.ok())
      return std::make_unique<RankedTensorType>(tensor.getShape(), *dtype);
  }

  // mlir::UnrankedTensorType -> xla::runtime::UnrankedTensorType
  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType());
        dtype.ok())
      return std::make_unique<UnrankedTensorType>(*dtype);
  }

  // mlir::MemrefType -> xla::runtime::MemrefType
  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType());
        dtype.ok())
      return std::make_unique<MemrefType>(memref.getShape(), *dtype);
  }

  // mlir::UnrankedMemrefType -> xla::runtime::UnrankedMemrefType
  if (auto memref = type.dyn_cast<mlir::UnrankedMemRefType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType());
        dtype.ok())
      return std::make_unique<UnrankedMemrefType>(*dtype);
  }

  // For non-canonical types the user must provide type conversion function.
  return {};
}

/*static*/ StatusOr<PrimitiveType> TypeConverter::ConvertElementType(
    mlir::Type type) {
  if (type.isF32()) return PrimitiveType::F32;
  if (type.isF64()) return PrimitiveType::F64;
  if (type.isUnsignedInteger(8)) return PrimitiveType::U8;
  if (type.isUnsignedInteger(16)) return PrimitiveType::U16;
  if (type.isUnsignedInteger(32)) return PrimitiveType::U32;
  if (type.isUnsignedInteger(64)) return PrimitiveType::U64;
  if (type.isInteger(1)) return PrimitiveType::PRED;
  if (type.isInteger(8)) return PrimitiveType::S8;
  if (type.isInteger(16)) return PrimitiveType::S16;
  if (type.isInteger(32)) return PrimitiveType::S32;
  if (type.isInteger(64)) return PrimitiveType::S64;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return PrimitiveType::C64;
    if (element_type.isF64()) return PrimitiveType::C128;
  }

  return InvalidArgumentError(
      StrFormat("unsupported element type: %s", ToString(type)));
}

StatusOr<std::unique_ptr<Type>> TypeConverter::Convert(mlir::Type type) const {
  if (auto converted = ConvertCanonicalType(type, *this)) return converted;

  for (const ConversionFn& conversion : conversions_)
    if (auto converted = conversion(type)) return converted;

  return InvalidArgumentError(
      StrFormat("can't convert type: %s to the run time type", ToString(type)));
}

StatusOr<FunctionType> TypeConverter::Convert(mlir::FunctionType type) const {
  assert(type && "function type must be not null");

  std::vector<std::unique_ptr<Type>> operands;
  std::vector<std::unique_ptr<Type>> results;

  operands.reserve(type.getNumInputs());
  results.reserve(type.getNumResults());

  auto error = [](std::string_view kind, unsigned i, mlir::Type type) {
    return InvalidArgumentError(
        StrFormat("can't convert %s #%i type %s to the run time type", kind, i,
                  ToString(type)));
  };

  for (unsigned i = 0; i < type.getNumInputs(); ++i) {
    StatusOr<std::unique_ptr<Type>> converted = Convert(type.getInput(i));
    if (!converted.ok()) return error("input", i, type.getInput(i));
    operands.push_back(std::move(*converted));
  }

  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    StatusOr<std::unique_ptr<Type>> converted = Convert(type.getResult(i));
    if (!converted.ok()) return error("result", i, type.getResult(i));
    results.push_back(std::move(*converted));
  }

  return FunctionType(std::move(operands), std::move(results));
}

}  // namespace runtime
}  // namespace xla
