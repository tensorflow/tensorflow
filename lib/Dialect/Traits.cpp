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
#include "llvm/Support/FormatVariadic.h"

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
  case StandardTypes::UnrankedTensor:
    return type.cast<TensorType>().getElementType().isIntOrFloat();
  default:
    break;
  }
  return false;
}

bool OpTrait::util::getBroadcastedShape(ArrayRef<int64_t> shape1,
                                        ArrayRef<int64_t> shape2,
                                        SmallVectorImpl<int64_t> &resultShape) {
  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.

  resultShape.clear();
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
    if (*i1 == -1 || *i2 == -1) {
      // One or both dimensions is unknown. Follow TensorFlow behavior:
      // - If either dimension is greater than 1, we assume that the program is
      //   correct, and the other dimension will be broadcast to match it.
      // - If either dimension is 1, the other dimension is the output.
      if (*i1 > 1) {
        *iR = *i1;
      } else if (*i2 > 1) {
        *iR = *i2;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else if (*i2 == 1) {
        *iR = *i1;
      } else {
        *iR = -1;
      }
    } else {
      if (*i1 == *i2 || *i2 == 1) {
        *iR = *i1;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else {
        // This dimension of the two operand types is incompatible.
        resultShape.clear();
        return false;
      }
    }
  }

  return true;
}

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
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

  // If one of the types is unranked tensor, then the other type shouldn't be
  // vector and the result should have unranked tensor type.
  if (type1.isa<UnrankedTensorType>() || type2.isa<UnrankedTensorType>()) {
    if (type1.isa<VectorType>() || type2.isa<VectorType>())
      return {};
    return UnrankedTensorType::get(scalarType);
  }

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
  SmallVector<int64_t, 4> resultShape;
  if (!getBroadcastedShape(getShape(type1), getShape(type2), resultShape))
    return {};

  // Compose the final broadcasted type
  if (resultCompositeKind == StandardTypes::Vector)
    return VectorType::get(resultShape, scalarType);
  if (resultCompositeKind == StandardTypes::RankedTensor)
    return RankedTensorType::get(resultShape, scalarType);
  return scalarType;
}

/// Returns true if the two given types are both vectors or ranked tensors and
/// they have the same shape, regardless of element types.
static bool isSameShapedVectorOrTensor(Type type1, Type type2) {
  if (auto vType1 = type1.dyn_cast<RankedTensorType>())
    if (auto vType2 = type2.dyn_cast<RankedTensorType>())
      return vType1.getShape() == vType2.getShape();
  if (auto vType1 = type1.dyn_cast<VectorType>())
    if (auto vType2 = type2.dyn_cast<VectorType>())
      return vType1.getShape() == vType2.getShape();
  return false;
}

bool OpTrait::impl::verifyCompatibleOperandBroadcast(const Instruction *op) {
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

  bool hasCompatRetType = (retType == broadcastedType) ||
                          retType.isa<UnrankedTensorType>() ||
                          isSameShapedVectorOrTensor(retType, broadcastedType);
  if (!hasCompatRetType)
    return op->emitOpError(
        llvm::formatv("result type '{0}' does not have the same shape as the "
                      "broadcasted type '{1}' computed from the operand types",
                      retType, broadcastedType));

  return false;
}
