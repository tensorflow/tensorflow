//===- Traits.cpp - Common op traits shared by dialects -------------------===//
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

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

/// Returns true if the given `type` supports NumPy broadcast semantics.
/// Specifically, the given `type` must be integer type, floating point type,
/// vector type, or ranked tensor type from integer or floating point types.
static bool isBroadcastableType(Type type) {
  switch (type.getKind()) {
  case StandardTypes::BF16:
  case StandardTypes::F16:
  case StandardTypes::F32:
  case StandardTypes::F64:
  case StandardTypes::Integer:
  case StandardTypes::Vector:
    return true;
  case StandardTypes::RankedTensor:
    return type.cast<RankedTensorType>().getElementType().isIntOrFloat();
  default:
    break;
  }
  return false;
}

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returns null type if the two given
/// types are not broadcast-compatible.
Type OpTrait::util::getBroadcastedType(Type type1, Type type2) {
  // Make sure both types are able to participate in broadcasting.
  if (!isBroadcastableType(type1) || !isBroadcastableType(type2))
    return {};

  // Returns the scalar type out of the given type.
  auto getScalarType = [](Type type) -> Type {
    if (auto vtType = type.dyn_cast<VectorOrTensorType>())
      return vtType.getElementType();
    return type;
  };

  // Make sure underlying scalar type is the same.
  auto scalarType = getScalarType(type1);
  if (scalarType != getScalarType(type2))
    return {};

  // Returns the type kind if the given type is a vector or ranked tensor type.
  // Returns llvm::None otherwise.
  auto getCompositeTypeKind =
      [](Type type) -> llvm::Optional<StandardTypes::Kind> {
    if (type.isa<VectorType>() || type.isa<RankedTensorType>())
      return static_cast<StandardTypes::Kind>(type.getKind());
    return llvm::None;
  };

  // Make sure the composite type, if has, is consistent.
  auto compositeKind1 = getCompositeTypeKind(type1);
  auto compositeKind2 = getCompositeTypeKind(type2);
  llvm::Optional<StandardTypes::Kind> resultCompositeKind;

  if (compositeKind1 && compositeKind2) {
    // Disallow mixing vector and tensor.
    if (compositeKind1 != compositeKind2)
      return {};
    resultCompositeKind = compositeKind1;
  } else if (compositeKind1) {
    resultCompositeKind = compositeKind1;
  } else if (compositeKind2) {
    resultCompositeKind = compositeKind2;
  }

  // Returns the shape of the given type.
  auto getShape = [](Type type) -> ArrayRef<int64_t> {
    if (auto vtType = type.dyn_cast<VectorOrTensorType>())
      return vtType.getShape();
    return {};
  };

  // Get the shape of each type.
  auto shape1 = getShape(type1);
  auto shape2 = getShape(type2);

  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  // 1. they are equal, or
  // 2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.

  SmallVector<int64_t, 4> resultShape;
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

  // Check each dimension is consistent.
  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    if (*i1 == *i2 || *i2 == 1) {
      *iR = *i1;
    } else if (*i1 == 1) {
      *iR = *i2;
    } else {
      // This dimension of the two operand types is incompatible.
      return {};
    }
  }

  // Compose the final broadcasted type
  if (resultCompositeKind == StandardTypes::Vector)
    return VectorType::get(resultShape, scalarType);
  if (resultCompositeKind == StandardTypes::RankedTensor)
    return RankedTensorType::get(resultShape, scalarType);
  return scalarType;
}

bool OpTrait::impl::verifyCompatibleOperandBroadcast(const OperationInst *op) {
  assert(op->getNumOperands() == 2 &&
         "only support broadcast check on two operands");
  assert(op->getNumResults() == 1 &&
         "only support broadcast check on one result");

  auto type1 = op->getOperand(0)->getType();
  auto type2 = op->getOperand(1)->getType();
  auto retType = op->getResult(0)->getType();

  auto broadcastedType = util::getBroadcastedType(type1, type2);

  if (!broadcastedType)
    return op->emitOpError("operands don't have broadcast-compatible types");
  if (broadcastedType != retType)
    return op->emitOpError(
        "result type is not broadcast-compatible with operand types");

  return false;
}
