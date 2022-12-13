/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ARITH_OPS_FOLDER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ARITH_OPS_FOLDER_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace mlir {

class Operation;

namespace TF {

class AddV2Op;
class SubOp;
class MulOp;
class DivOp;
class RealDivOp;

// Verifies an reduction op's `input` and reduction `dims`.
LogicalResult VerifyReductionInputAndDims(Value input, Value dims,
                                          Location loc);

// A type range with description (in singular form) attached to it.
using TypeRangeWithDesc = std::pair<TypeRange, StringRef>;

LogicalResult VerifyTypeRangesAreCompatible(Operation *op,
                                            TypeRangeWithDesc range0,
                                            TypeRangeWithDesc range1);

// Fold Arithmetic Op if one of the operands is a constant known to be an
// Identity (e.g. X+0, X*1, etc...). For commutative operations fold if
// known identity value is either lhs or rhs.
template <
    typename OpT,
    typename std::enable_if<llvm::is_one_of<
        OpT, AddV2Op, SubOp, MulOp, DivOp, RealDivOp>::value>::type * = nullptr>
OpFoldResult IdentityArithmeticOpFolder(OpT arithmetic_op,
                                        ArrayRef<Attribute> operands) {
  auto lhs_type = arithmetic_op.getX().getType().template cast<ShapedType>();
  auto rhs_type = arithmetic_op.getY().getType().template cast<ShapedType>();
  auto result_type =
      arithmetic_op.getResult().getType().template cast<ShapedType>();

  // We can fold arithmetic operation only of we can prove that we will not
  // accidentally hide a broadcasting error.
  auto is_valid_broadcasting = [](ShapedType operand_ty, ShapedType identity_ty,
                                  ShapedType result_ty) -> bool {
    // Scalar identity is broadcastable to any operand shape, we only need to
    // check that operand has the same shape as a result.
    bool scalar_identity = identity_ty.hasRank() && identity_ty.getRank() == 0;
    if (scalar_identity) return operand_ty == result_ty;

    // If identity is not a scalar, we must verify that identity shape is
    // statically known to be broadcastable to the operand shape and the operand
    // and result shape are equal.
    return operand_ty == result_ty && identity_ty.hasStaticShape() &&
           result_ty.hasStaticShape() &&
           OpTrait::util::staticallyKnownBroadcastable(operand_ty.getShape(),
                                                       identity_ty.getShape());
  };

  // Check that we have a constant operand on one side (candidate for identity).
  const bool is_commutative =
      (std::is_same<OpT, AddV2Op>::value || std::is_same<OpT, MulOp>::value);
  auto lhs_attr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhs_attr = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!rhs_attr && !(is_commutative && lhs_attr)) return {};

  // Mul and Div ops have identity value one while AddV2 and SubOp have identity
  // value zero.
  const int identity =
      (std::is_same<OpT, MulOp>::value || std::is_same<OpT, DivOp>::value ||
       std::is_same<OpT, RealDivOp>::value)
          ? 1
          : 0;

  Type element_ty = lhs_type.getElementType();
  Attribute identity_attr;
  if (auto ty = element_ty.template dyn_cast<FloatType>()) {
    identity_attr = FloatAttr::get(ty, static_cast<double>(identity));
  } else if (auto ty = element_ty.template dyn_cast<IntegerType>()) {
    identity_attr = IntegerAttr::get(ty, static_cast<int64_t>(identity));
  } else {
    return {};
  }

  // Fold: Op(Operand, Identity) -> Operand.
  if (rhs_attr && is_valid_broadcasting(lhs_type, rhs_type, result_type)) {
    if (rhs_attr.isSplat() &&
        rhs_attr.getSplatValue<Attribute>() == identity_attr)
      return arithmetic_op.getX();
  }

  // Fold: Op(Identity, Operand) -> Operand for commutative operations.
  if (lhs_attr && is_commutative &&
      is_valid_broadcasting(rhs_type, lhs_type, result_type)) {
    if (lhs_attr.isSplat() &&
        lhs_attr.getSplatValue<Attribute>() == identity_attr)
      return arithmetic_op.getY();
  }

  return {};
}

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ARITH_OPS_FOLDER_H_
