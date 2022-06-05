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

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_H
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_H

#include "llvm/ADT/Sequence.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace mhlo {

// Forward declaration for a function declared in hlo_ops.h.
bool isCompatibleForMhloTypeInference(Type tp1, Type tp2);

namespace OpTrait {

template <typename ConcreteType>
class BroadcastingElementwise
    : public mlir::OpTrait::TraitBase<ConcreteType, BroadcastingElementwise> {};

template <typename ConcreteType>
class PairwiseSameOperandAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      PairwiseSameOperandAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    const int numOperands = op->getNumOperands();
    const int numResults = op->getNumResults();
    if (numOperands != numResults) {
      return op->emitOpError()
             << "requires the same number of operands and results";
    }

    for (int idx : llvm::seq<int>(0, numOperands)) {
      if (op->getOperand(idx).getType() != op->getResult(idx).getType()) {
        return op->emitOpError()
               << "requires the same type for operand and result at index "
               << idx;
      }
    }
    return success();
  }
};

template <typename ConcreteType>
class CompatibleOperandsAndResultType
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      CompatibleOperandsAndResultType> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    Type expected;
    if (op->getNumResults() != 0) expected = op->getResult(0).getType();
    if (op->getNumOperands() != 0) expected = op->getOperand(0).getType();
    if (!expected) return failure();

    auto type_match = [&](Type actual) {
      return isCompatibleForMhloTypeInference(actual, expected);
    };
    auto all_match = llvm::all_of(op->getOperandTypes(), type_match) &&
                     llvm::all_of(op->getResultTypes(), type_match);
    if (!all_match) {
      return op->emitOpError(
          "requires compatible types for all operands and results");
    }

    return success(all_match);
  }

  static LogicalResult inferReturnTypes(
      MLIRContext * /*context*/, Optional<Location> location,
      ValueRange operands, DictionaryAttr /*attributes*/,
      RegionRange /*regions*/, SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
    // support quantization or sparsity.
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");
    inferredReturnTypes.push_back(operands[0].getType());
    return success();
  }

  // This function is not going to be called automatically.
  // It needs to be paired with INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS
  // (see examples in hlo_ops.cc).
  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext *context, Optional<Location> location,
      ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<Type> inferred_return_types;
    if (failed(inferReturnTypes(context, location, operands.getValues(),
                                attributes, regions, inferred_return_types)))
      return failure();
    auto inferred_return_type = inferred_return_types[0].cast<ShapedType>();
    inferredReturnShapes.push_back(inferred_return_type);
    return success();
  }
};

}  // namespace OpTrait
}  // namespace mhlo
}  // namespace mlir

#endif
