//===- Traits.cpp - Common op traits shared by dialects -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

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

/// Returns the shape of the given type. Scalars will be considered as having a
/// shape with zero dimensions.
static ArrayRef<int64_t> getShape(Type type) {
  if (auto sType = type.dyn_cast<ShapedType>())
    return sType.getShape();
  return {};
}

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
Type OpTrait::util::getBroadcastedType(Type type1, Type type2) {
  // Returns the scalar type out of the given type.
  auto getScalarType = [](Type type) -> Type {
    if (auto shapedType = type.dyn_cast<ShapedType>())
      return shapedType.getElementType();
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
  auto getCompositeTypeKind = [](Type type) -> Optional<StandardTypes::Kind> {
    if (type.isa<VectorType>() || type.isa<RankedTensorType>())
      return static_cast<StandardTypes::Kind>(type.getKind());
    return llvm::None;
  };

  // Make sure the composite type, if has, is consistent.
  auto compositeKind1 = getCompositeTypeKind(type1);
  auto compositeKind2 = getCompositeTypeKind(type2);
  Optional<StandardTypes::Kind> resultCompositeKind;

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

/// Returns true if the given types has both vector types and tensor types.
static bool hasBothVectorAndTensorType(ArrayRef<Type> types) {
  return llvm::any_of(types, [](Type t) { return t.isa<VectorType>(); }) &&
         llvm::any_of(types, [](Type t) { return t.isa<TensorType>(); });
}

static bool areCompatibleShapes(ArrayRef<int64_t> shape1,
                                ArrayRef<int64_t> shape2) {
  auto isCompatible = [](int64_t dim1, int64_t dim2) {
    return dim1 == dim2 || dim1 == -1 || dim2 == -1;
  };
  if (shape1.size() != shape2.size())
    return false;
  for (const auto &p : llvm::zip(shape1, shape2))
    if (!isCompatible(std::get<0>(p), std::get<1>(p)))
      return false;
  return true;
}

LogicalResult OpTrait::impl::verifyCompatibleOperandBroadcast(Operation *op) {
  assert(op->getNumOperands() == 2 &&
         "only support broadcast check on two operands");
  assert(op->getNumResults() == 1 &&
         "only support broadcast check on one result");

  auto type1 = op->getOperand(0)->getType();
  auto type2 = op->getOperand(1)->getType();
  auto retType = op->getResult(0)->getType();

  // We forbid broadcasting vector and tensor.
  if (hasBothVectorAndTensorType({type1, type2, retType}))
    return op->emitError("cannot broadcast vector with tensor");

  if (retType.isa<UnrankedTensorType>())
    return success();

  bool isUnranked1 = type1.isa<UnrankedTensorType>();
  bool isUnranked2 = type2.isa<UnrankedTensorType>();

  // If both operands are unranked, then all result shapes are possible.
  if (isUnranked1 && isUnranked2)
    return success();

  // If one of the operands is unranked, then the known dimensions in the result
  // should be compatible with the other shaped operand.
  if (isUnranked1 || isUnranked2) {
    // Result should have higher rank than the shaped operand's rank and then
    // the result's trailing dimensions should be compatible with the operand
    // shape.
    ArrayRef<int64_t> shape = getShape(!isUnranked1 ? type1 : type2);
    ArrayRef<int64_t> actualSuffix = getShape(retType).take_back(shape.size());
    if (!areCompatibleShapes(actualSuffix, shape))
      return op->emitOpError()
             << "result type " << retType
             << " has shape incompatible with a ranked operand type";
    return success();
  }

  // If both operands are shaped, then the computed broadcasted shape should be
  // compatible with the result shape.
  SmallVector<int64_t, 4> resultShape;
  if (!util::getBroadcastedShape(getShape(type1), getShape(type2), resultShape))
    return op->emitOpError("operands don't have broadcast-compatible shapes");

  if (!areCompatibleShapes(resultShape, getShape(retType)))
    return op->emitOpError() << "result type " << retType
                             << " does not have shape compatible with the one "
                                "computed from the operand types";

  return success();
}
