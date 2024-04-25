/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/runtime/transforms/type_converter.h"

#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_dialect.h"
#include "xla/primitive_util.h"
#include "xla/runtime/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace runtime {

using absl::InvalidArgumentError;
using absl::StatusOr;
using absl::StrFormat;

// Type conversion for the canonical MLIR types supported by the runtime.
static std::unique_ptr<Type> ConvertCanonicalType(
    mlir::Type type, const TypeConverter& convert) {
  // ExecutionContextType -> ExecutionContextOperandType (both in xla::runtime).
  if (auto ctx = mlir::dyn_cast<ExecutionContextType>(type))
    return std::make_unique<ExecutionContextOperandType>();

  // OpaqueType -> OpaqueOperandType (both in xla::runtime).
  if (auto ctx = mlir::dyn_cast<OpaqueType>(type))
    return std::make_unique<OpaqueOperandType>();

  // mlir::async::TokenType -> xla::runtime::AsyncTokenType
  if (mlir::isa<mlir::async::TokenType>(type))
    return std::make_unique<AsyncTokenType>();

  // mlir::async::ValueType -> xla::runtime::AsyncValueType
  if (auto value = mlir::dyn_cast<mlir::async::ValueType>(type)) {
    if (auto value_type = convert.Convert(value.getValueType());
        value_type.ok())
      return std::make_unique<AsyncValueType>(std::move(*value_type));
  }

  // mlir::{IndexType, IntegerType, FloatType} -> xla::runtime::ScalarType
  if (mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(type)) {
    if (auto dtype = TypeConverter::ConvertElementType(type); dtype.ok())
      return std::make_unique<ScalarType>(*dtype);
  }

  // mlir::RankedTensorType -> xla::runtime::RankedTensorType
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType());
        dtype.ok())
      return std::make_unique<RankedTensorType>(tensor.getShape(), *dtype);
  }

  // mlir::UnrankedTensorType -> xla::runtime::UnrankedTensorType
  if (auto tensor = mlir::dyn_cast<mlir::UnrankedTensorType>(type)) {
    if (auto dtype = TypeConverter::ConvertElementType(tensor.getElementType());
        dtype.ok())
      return std::make_unique<UnrankedTensorType>(*dtype);
  }

  // mlir::MemrefType -> xla::runtime::MemrefType
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType());
        dtype.ok())
      return std::make_unique<MemrefType>(memref.getShape(), *dtype);
  }

  // mlir::UnrankedMemrefType -> xla::runtime::UnrankedMemrefType
  if (auto memref = mlir::dyn_cast<mlir::UnrankedMemRefType>(type)) {
    if (auto dtype = TypeConverter::ConvertElementType(memref.getElementType());
        dtype.ok())
      return std::make_unique<UnrankedMemrefType>(*dtype);
  }

  // mlir::TupleType -> xla::runtime::TupleType
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type)) {
    llvm::SmallVector<std::unique_ptr<Type>> conv_elems;
    llvm::transform(tuple, std::back_inserter(conv_elems),
                    [&convert](mlir::Type type) {
                      return ConvertCanonicalType(type, convert);
                    });
    return std::make_unique<TupleType>(std::move(conv_elems));
  }

  // For non-canonical types the user must provide type conversion function.
  return {};
}

/*static*/ StatusOr<PrimitiveType> TypeConverter::ConvertElementType(
    mlir::Type type) {
  if (type.isFloat8E4M3FN()) return PrimitiveType::F8E4M3FN;
  if (type.isFloat8E4M3B11FNUZ()) return PrimitiveType::F8E4M3B11FNUZ;
  if (type.isFloat8E4M3FNUZ()) return PrimitiveType::F8E4M3FNUZ;
  if (type.isFloat8E5M2()) return PrimitiveType::F8E5M2;
  if (type.isFloat8E5M2FNUZ()) return PrimitiveType::F8E5M2FNUZ;
  if (type.isIndex()) return PrimitiveType::S64;
  if (type.isBF16()) return PrimitiveType::BF16;
  if (type.isF16()) return PrimitiveType::F16;
  if (type.isF32()) return PrimitiveType::F32;
  if (type.isF64()) return PrimitiveType::F64;
  if (type.isInteger(1)) return PrimitiveType::PRED;
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    unsigned int width = int_type.getWidth();
    if (auto primitive_type =
            int_type.isUnsigned()
                ? primitive_util::UnsignedIntegralTypeForBitWidth(width)
                : primitive_util::SignedIntegralTypeForBitWidth(width);
        primitive_type != PRIMITIVE_TYPE_INVALID) {
      return primitive_type;
    }
  }
  if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    auto element_type = complex_type.getElementType();
    TF_ASSIGN_OR_RETURN(auto element_primitive_type,
                        ConvertElementType(element_type));
    if (auto complex_primitive_type =
            primitive_util::ComplexType(element_primitive_type);
        complex_primitive_type != PRIMITIVE_TYPE_INVALID) {
      return complex_primitive_type;
    }
  }

  return InvalidArgumentError(
      StrFormat("unsupported element type: %s", debugString(type)));
}

StatusOr<std::unique_ptr<Type>> TypeConverter::Convert(mlir::Type type) const {
  if (std::unique_ptr<Type> converted = ConvertCanonicalType(type, *this))
    return std::move(converted);

  for (const ConversionFn& conversion : conversions_)
    if (std::unique_ptr<Type> converted = conversion(type))
      return std::move(converted);

  return InvalidArgumentError(StrFormat(
      "can't convert type: %s to the run time type", debugString(type)));
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
                  debugString(type)));
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
