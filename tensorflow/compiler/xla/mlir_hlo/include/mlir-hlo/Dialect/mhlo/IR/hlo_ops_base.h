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

#include <algorithm>

#include "llvm/ADT/Sequence.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Include order below matters.
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h.inc"

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

    auto typeMatch = [&](Type actual) {
      return isCompatibleForMhloTypeInference(actual, expected);
    };
    auto allMatch = llvm::all_of(op->getOperandTypes(), typeMatch) &&
                    llvm::all_of(op->getResultTypes(), typeMatch);
    if (!allMatch) {
      return op->emitOpError(
          "requires compatible types for all operands and results");
    }

    return success(allMatch);
  }

  static LogicalResult inferReturnTypes(
      MLIRContext *context, Optional<Location> location, ValueRange operands,
      DictionaryAttr /*attributes*/, RegionRange /*regions*/,
      SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
    // support quantization or sparsity.
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");

    if (failed(inferMostSpecificType(context, location, operands.getTypes(),
                                     inferredReturnTypes)))
      return failure();
    return success();
  }

  // This function is not going to be called automatically.
  // It needs to be paired with INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS
  // (see examples in hlo_ops.cc).
  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext *context, Optional<Location> location,
      ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<Type> inferredReturnTypes;
    if (failed(inferReturnTypes(context, location, operands.getValues(),
                                attributes, regions, inferredReturnTypes)))
      return failure();
    auto inferredReturnType = inferredReturnTypes[0].cast<ShapedType>();
    inferredReturnShapes.push_back(inferredReturnType);
    return success();
  }

 private:
  // Cases of infer return shape with bounds (lhs and rhs are commutative):
  //       Dim of lhs     Dim of rhs      Infer
  //  c0:  3              3               3
  //  c1:  3              ?               3
  //  c2:  3              ?, bound=4      3
  //  c3:  3              ?, bound=2      Error out
  //  c4:  ?              ?               ?
  //  c5:  ?              ?, bound=3      ?, bound=3
  //  c6:  ?, bound=3     ?, bound=3      ?, bound=3
  //  c7:  ?, bound=3     ?, bound=4      ?, bound=3
  // This method generalizes it to multiple inputs: 1) get the static input dims
  // (if any) as infer dim, and 2) get min of input bounds as infer bound
  static LogicalResult inferMostSpecificType(
      MLIRContext *context, Optional<Location> location,
      ValueTypeRange<ValueRange> inputTypes,
      SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(zhouxin) remove this part and find a way to infer sparsity encoding.
    if (inputTypes.size() == 1) {
      inferredReturnTypes.push_back(inputTypes[0]);
      return success();
    }

    SmallVector<RankedTensorType> rankedTypes;
    for (auto inputType : inputTypes)
      if (auto rankedType = inputType.dyn_cast<RankedTensorType>())
        rankedTypes.push_back(rankedType);
    if (rankedTypes.empty()) {
      inferredReturnTypes.push_back(inputTypes[0]);
      return success();
    }

    auto rank = rankedTypes[0].getRank();
    SmallVector<int64_t> inferredDimSizes(rank, ShapedType::kDynamicSize);
    SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamicSize);
    for (auto rankedType : rankedTypes) {
      SmallVector<int64_t> bounds;
      if (auto encoding =
              rankedType.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>())
        bounds = llvm::to_vector<4>(encoding.getBounds());

      for (int dim = 0; dim < rank; ++dim) {
        // Dimensions
        auto dimSize = rankedType.getShape()[dim];
        if (inferredDimSizes[dim] != ShapedType::kDynamicSize &&
            dimSize != ShapedType::kDynamicSize &&
            inferredDimSizes[dim] != dimSize)
          return emitOptionalError(location, "Mismatch dimension size ",
                                   inferredDimSizes[dim], " and ", dimSize,
                                   " in dimension ", dim);
        if (inferredDimSizes[dim] == ShapedType::kDynamicSize)
          inferredDimSizes[dim] = dimSize;

        // Bounds
        if (!bounds.empty() && bounds[dim] != ShapedType::kDynamicSize) {
          if (inferredBounds[dim] == ShapedType::kDynamicSize) {
            inferredBounds[dim] = bounds[dim];
          } else {
            inferredBounds[dim] = std::min(inferredBounds[dim], bounds[dim]);
          }
        }
        // Error out case that the inferred bound is smaller than inferred dim
        if (inferredBounds[dim] != ShapedType::kDynamicSize &&
            inferredBounds[dim] < inferredDimSizes[dim])
          return emitOptionalError(location,
                                   "bound must not be less than static "
                                   "dimension size but has bound ",
                                   inferredBounds[dim], " vs static size ",
                                   inferredDimSizes[dim], " in dimension ",
                                   dim);
        if (inferredDimSizes[dim] != ShapedType::kDynamicSize)
          inferredBounds[dim] = ShapedType::kDynamicSize;
      }
    }

    Attribute encoding = nullptr;
    if (llvm::any_of(inferredBounds,
                     [](auto el) { return el != ShapedType::kDynamicSize; }))
      encoding = TypeExtensionsAttr::get(context, inferredBounds);
    inferredReturnTypes.push_back(RankedTensorType::get(
        inferredDimSizes, rankedTypes[0].getElementType(), encoding));

    return success();
  }
};

}  // namespace OpTrait
}  // namespace mhlo
}  // namespace mlir

#endif
