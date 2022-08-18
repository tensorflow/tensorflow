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
#include <utility>

#include "mlir/Dialect/Async/IR/AsyncTypes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace xla {
namespace runtime {

using llvm::Expected;

using tfrt::DType;
using tfrt::MakeStringError;

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
    if (auto value_type = convert.Convert(value.getValueType()))
      return std::make_unique<AsyncValueType>(std::move(*value_type));
  }

  // mlir::RankedTensorType -> xla::runtime::RankedTensorType
  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType()))
      return std::make_unique<RankedTensorType>(tensor.getShape(), *dtype);
  }

  // mlir::UnrankedTensorType -> xla::runtime::UnrankedTensorType
  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType()))
      return std::make_unique<UnrankedTensorType>(*dtype);
  }

  // mlir::MemrefType -> xla::runtime::MemrefType
  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType()))
      return std::make_unique<MemrefType>(memref.getShape(), *dtype);
  }

  // mlir::UnrankedMemrefType -> xla::runtime::UnrankedMemrefType
  if (auto memref = type.dyn_cast<mlir::UnrankedMemRefType>()) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType()))
      return std::make_unique<UnrankedMemrefType>(*dtype);
  }

  // For non-canonical types the user must provide type conversion function.
  return {};
}

/*static*/ Expected<DType> TypeConverter::ConvertElementType(mlir::Type type) {
  if (type.isF32()) return DType::F32;
  if (type.isF64()) return DType::F64;
  if (type.isUnsignedInteger(8)) return DType::UI8;
  if (type.isUnsignedInteger(16)) return DType::UI16;
  if (type.isUnsignedInteger(32)) return DType::UI32;
  if (type.isUnsignedInteger(64)) return DType::UI64;
  if (type.isInteger(1)) return DType::I1;
  if (type.isInteger(8)) return DType::I8;
  if (type.isInteger(16)) return DType::I16;
  if (type.isInteger(32)) return DType::I32;
  if (type.isInteger(64)) return DType::I64;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return DType::Complex64;
    if (element_type.isF64()) return DType::Complex128;
  }

  return MakeStringError("unsupported element type: ", type);
}

Expected<std::unique_ptr<Type>> TypeConverter::Convert(mlir::Type type) const {
  if (auto converted = ConvertCanonicalType(type, *this)) return converted;

  for (const ConversionFn& conversion : conversions_)
    if (auto converted = conversion(type)) return converted;

  return MakeStringError("can't convert type: ", type, " to the run time type");
}

Expected<FunctionType> TypeConverter::Convert(mlir::FunctionType type) const {
  assert(type && "function type must be not null");

  llvm::SmallVector<std::unique_ptr<Type>> operands;
  llvm::SmallVector<std::unique_ptr<Type>> results;

  operands.reserve(type.getNumInputs());
  results.reserve(type.getNumResults());

  auto error = [](llvm::StringRef kind, unsigned i, mlir::Type type) {
    return MakeStringError("can't convert ", kind, " #", i, " type ", type,
                           " to the run time type");
  };

  for (unsigned i = 0; i < type.getNumInputs(); ++i) {
    Expected<std::unique_ptr<Type>> converted = Convert(type.getInput(i));
    if (!converted) return error("input", i, type.getInput(i));
    operands.push_back(std::move(*converted));
  }

  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    Expected<std::unique_ptr<Type>> converted = Convert(type.getResult(i));
    if (!converted) return error("result", i, type.getResult(i));
    results.push_back(std::move(*converted));
  }

  return FunctionType(std::move(operands), std::move(results));
}

}  // namespace runtime
}  // namespace xla
