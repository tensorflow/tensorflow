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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_ATTRS_AND_CONSTRAINTS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_ATTRS_AND_CONSTRAINTS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/status/statusor.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"

namespace mlir::quant {

constexpr char kAttrMapAttribute[] = "attr_map";

// Permutation from the NHWC tensor format to NCHW. This is an inverse
// permutation of `kNchwToNhwcPermutation`.
inline constexpr std::array<int64_t, 4> kNhwcToNchwPermutation = {0, 3, 1, 2};

// Permutation from the NCHW tensor format to NHWC. This is an inverse
// permutation of `kNchwToNhwcPermutation`.
inline constexpr std::array<int64_t, 4> kNchwToNhwcPermutation = {0, 2, 3, 1};

// Permutation from the OIHW (== (output features, input features, height,
// width)) tensor format to HWIO. This is commonly used to transpose convolution
// weights represented as OIHW format to HWIO, which is more desirable for
// certain downstream optimization passes (e.g. XLA).
inline constexpr std::array<int64_t, 4> kOihwToHwioPermutation = {2, 3, 1, 0};

// Returns true if the value has static shape.
bool HasStaticShape(Value value);

// Returns true if the value has static shape at given dims.
bool HasStaticShapeAtDims(Value value, ArrayRef<int> dims);

// Whether `value` has known rank of `rank`. Returns false when it is not a
// `ShapedType` or its rank is unknown.
inline bool HasRankOf(Value value, const int64_t rank) {
  auto shaped_type = value.getType().dyn_cast_or_null<ShapedType>();
  return shaped_type && shaped_type.hasRank() && shaped_type.getRank() == rank;
}

// Creates a new type that has the shape from the `old_type` and the element
// type from the `element_type`.
Type CloneTypeWithNewElementType(Type old_type, Type element_type);

// Creates an array with integer/float type.
template <typename T,
          typename = std::enable_if_t<
              (std::is_integral_v<T> || std::is_same_v<T, float>), void>>
Value CreateConstValue(OpBuilder& builder, const Location loc,
                       const SmallVector<int64_t>& shape,
                       const SmallVector<T>& values) {
  if constexpr (std::is_integral_v<T>) {
    auto shape_type =
        RankedTensorType::get(shape, builder.getIntegerType(sizeof(T) * 8));

    const auto attr = DenseIntElementsAttr::get(shape_type, values);
    return builder.create<TF::ConstOp>(loc, attr);
  }

  const auto type = RankedTensorType::get(shape, builder.getF32Type());
  const auto value_attr = DenseFPElementsAttr::get(type, values);
  return builder.create<TF::ConstOp>(loc, value_attr);
}

// Creates a 1D array with integer/float type.
template <typename T>
Value Create1DConstValue(OpBuilder& builder, const Location loc,
                         const SmallVector<T>& values) {
  return CreateConstValue<T>(builder, loc,
                             {static_cast<int64_t>(values.size())}, values);
}

// Creates a scalar with integer / float type.
template <typename T>
Value CreateScalarConstValue(OpBuilder& builder, const Location loc,
                             const T value) {
  return CreateConstValue<T>(builder, loc, /*shape=*/{}, {value});
}

// Checks if the value is a constant and return its splat value.
template <typename T,
          typename = std::enable_if_t<
              (std::is_integral_v<T> || std::is_same_v<T, float>), void>>
bool GetSplatValue(Value value, T& splat_value) {
  if constexpr (std::is_integral_v<T>) {
    DenseIntElementsAttr value_attr;
    if (!matchPattern(value, m_Constant(&value_attr)) ||
        !value_attr.isSplat()) {
      return false;
    }
    splat_value = value_attr.getSplatValue<T>();
    return true;
  }

  DenseFPElementsAttr value_attr;
  if (!matchPattern(value, m_Constant(&value_attr)) || !value_attr.isSplat()) {
    return false;
  }
  splat_value = value_attr.getSplatValue<T>();
  return true;
}

// Checks if the value is a constant and its splat value is equal to x.
template <typename T>
bool IsSplatValueEqual(Value value, const T x) {
  T splat_value;
  if (!GetSplatValue(value, splat_value)) return false;

  return splat_value == x;
}

// Checks if two values are constants and their splat values are equal.
template <typename T>
bool AreSplatValuesEqual(Value x, Value y) {
  T splat_x, splat_y;
  if (!GetSplatValue(x, splat_x) || !GetSplatValue(y, splat_y)) {
    return false;
  }

  return splat_x == splat_y;
}

// Clones an operation with new operands while keeping attributes.
SmallVector<Value> CloneOpWithReplacedOperands(OpBuilder& builder,
                                               Operation* op,
                                               ArrayRef<Value> new_operands);

// Tries casting `op` with a concrete op type `T`. If the cast fails or `op` is
// a `nullptr`, returns `failure` and prints a debugging message identifying
// the cast attempt as `name`.
template <typename T>
FailureOr<T> TryCast(Operation* op, const StringRef name) {
  auto cast_op = dyn_cast_or_null<T>(op);
  if (cast_op) {
    return cast_op;
  } else {
    DEBUG_WITH_TYPE("mlir-quant-attrs-and-constraints",
                    llvm::dbgs() << "Failed to match " << name << " ("
                                 << T::getOperationName() << ").\n");
    return failure();
  }
}

FailureOr<int32_t> CastI64ToI32(int64_t value);

// Tries to cast an array of int64 to int32. If any of the element in the
// array is not in the range of int32, returns failure().
FailureOr<SmallVector<int32_t>> CastI64ArrayToI32(
    ArrayRef<int64_t> int64_array);

// Returns the first operation with the given type in the function.
template <typename OpType>
OpType FindOperationOfType(func::FuncOp function) {
  for (auto op : function.getBody().getOps<OpType>()) {
    return op;
  }
  return nullptr;
}

// Returns the first user of the given operation, optionally of the given
// type if provided. If there is no user or user of type, return nullptr.
template <typename T = Operation*>
Operation* FindUserOfType(Operation* op) {
  for (Operation* user : op->getUsers()) {
    if (isa<T>(user)) {
      return user;
    }
  }
  return nullptr;
}

// Returns the first user of the given operation, optionally of the given
// type if provided. If there is no user or user of type, return nullptr.
template <typename T = Operation*>
Operation* FindOperandOfType(Operation* op) {
  for (Value operand_value : op->getOperands()) {
    if (isa<T>(operand_value.getDefiningOp())) {
      return operand_value.getDefiningOp();
    }
  }
  return nullptr;
}

// Returns the function attribute for the given call op which is lifted for
// quantization.
inline FlatSymbolRefAttr GetFuncAttr(TF::PartitionedCallOp call_op) {
  return call_op.getFAttr().template dyn_cast<FlatSymbolRefAttr>();
}

inline FlatSymbolRefAttr GetFuncAttr(TF::XlaCallModuleOp call_op) {
  return call_op->getAttrOfType<FlatSymbolRefAttr>(
      TF::kStablehloEntryFunctionAttrName);
}

// Returns the entry function name for the given tf.XlaCallModule op. Returns
// empty string if such attribute does not exist.
StringRef GetEntryFunctionName(TF::XlaCallModuleOp op);

// Checks whether the given op contains QuantizationTrait::FullyQuantizable.
inline bool HasQuantizableTrait(Operation* op) {
  return op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
         op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
             QuantTraitValues[QuantizationTrait::FullyQuantizable];
}

// Returns true if `op` has two operands and one result and only second operand
// is quantized.
bool IsHybridQuantizedOp(Operation* op);

// Returns whether a given `stablehlo.dot_general` can be legalizable to
// `tfl.fully_connected`.
absl::StatusOr<bool> IsDotGeneralFullyConnected(
    ::mlir::stablehlo::DotGeneralOp dot_general_op);

// Returns the quantization dimension for a given `stablehlo.dot_general` op,
// or `std::nullopt` if the given op is not per-channel quantizable.
std::optional<int64_t> GetDotGeneralQuantizationDim(
    ::mlir::stablehlo::DotGeneralOp dot_general_op);

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_ATTRS_AND_CONSTRAINTS_H_
