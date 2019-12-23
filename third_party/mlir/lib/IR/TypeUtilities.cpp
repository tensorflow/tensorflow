//===- TypeUtilities.cpp - Helper function for type queries ---------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines generic type utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;

Type mlir::getElementTypeOrSelf(Type type) {
  if (auto st = type.dyn_cast<ShapedType>())
    return st.getElementType();
  return type;
}

Type mlir::getElementTypeOrSelf(Value val) {
  return getElementTypeOrSelf(val->getType());
}

Type mlir::getElementTypeOrSelf(Attribute attr) {
  return getElementTypeOrSelf(attr.getType());
}

SmallVector<Type, 10> mlir::getFlattenedTypes(TupleType t) {
  SmallVector<Type, 10> fTypes;
  t.getFlattenedTypes(fTypes);
  return fTypes;
}

/// Return true if the specified type is an opaque type with the specified
/// dialect and typeData.
bool mlir::isOpaqueTypeWithName(Type type, StringRef dialect,
                                StringRef typeData) {
  if (auto opaque = type.dyn_cast<mlir::OpaqueType>())
    return opaque.getDialectNamespace().is(dialect) &&
           opaque.getTypeData() == typeData;
  return false;
}

/// Returns success if the given two shapes are compatible. That is, they have
/// the same size and each pair of the elements are equal or one of them is
/// dynamic.
LogicalResult mlir::verifyCompatibleShape(ArrayRef<int64_t> shape1,
                                          ArrayRef<int64_t> shape2) {
  if (shape1.size() != shape2.size())
    return failure();
  for (const auto &dims : llvm::zip(shape1, shape2)) {
    int64_t dim1 = std::get<0>(dims);
    int64_t dim2 = std::get<1>(dims);
    if (!ShapedType::isDynamic(dim1) && !ShapedType::isDynamic(dim2) &&
        dim1 != dim2)
      return failure();
  }
  return success();
}

/// Returns success if the given two types have compatible shape. That is,
/// they are both scalars (not shaped), or they are both shaped types and at
/// least one is unranked or they have compatible dimensions. Dimensions are
/// compatible if at least one is dynamic or both are equal. The element type
/// does not matter.
LogicalResult mlir::verifyCompatibleShape(Type type1, Type type2) {
  auto sType1 = type1.dyn_cast<ShapedType>();
  auto sType2 = type2.dyn_cast<ShapedType>();

  // Either both or neither type should be shaped.
  if (!sType1)
    return success(!sType2);
  if (!sType2)
    return failure();

  if (!sType1.hasRank() || !sType2.hasRank())
    return success();

  return verifyCompatibleShape(sType1.getShape(), sType2.getShape());
}

OperandElementTypeIterator::OperandElementTypeIterator(
    Operation::operand_iterator it)
    : llvm::mapped_iterator<Operation::operand_iterator, Type (*)(Value)>(
          it, &unwrap) {}

Type OperandElementTypeIterator::unwrap(Value value) {
  return value->getType().cast<ShapedType>().getElementType();
}

ResultElementTypeIterator::ResultElementTypeIterator(
    Operation::result_iterator it)
    : llvm::mapped_iterator<Operation::result_iterator, Type (*)(Value)>(
          it, &unwrap) {}

Type ResultElementTypeIterator::unwrap(Value value) {
  return value->getType().cast<ShapedType>().getElementType();
}
