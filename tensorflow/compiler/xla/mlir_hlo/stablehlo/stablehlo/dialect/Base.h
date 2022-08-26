/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_BASE_H
#define STABLEHLO_DIALECT_BASE_H

#include <algorithm>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"

// Include order matters
#include "stablehlo/dialect/BaseAttrInterfaces.h.inc"

namespace mlir {
namespace hlo {

// Returns true if the given types are the same for the purposes of HLO type
// inference, accounting for special properties of quantization and sparsity.
bool isCompatibleForHloTypeInference(Type tp1, Type tp2);

// Shape derivation function that computes the shape of the result based on an
// operand. For a 2-dimensional input tensor, this produces IR of the form
//
//  %0 = dim %arg0, 0 : memref<?x?xf32>
//  %1 = index_cast %0 : index to i64
//  %2 = dim %arg0, 1 : memref<?x?xf32>
//  %3 = index_cast %2 : index to i64
//  %4 = "shape.shape_of"(%1, %3)
//    : (i64, i64) -> tensor<2xi64>
//
// and returns %4 as the shape value.
LogicalResult deriveShapeFromOperand(
    OpBuilder *builder, Operation *op, Value operand,
    SmallVectorImpl<Value> *reifiedReturnShapes);

// Type derivation function that returns a tensor type with a new element type.
TensorType getSameShapeTensorType(TensorType tensorType, Type elementType);

// Verify bounds expressed by HLO_BoundedInterface against the provided type.
// See documentation for HLO_BoundedInterface for the list of checks.
LogicalResult verifyBounds(ArrayRef<int64_t> bounds, ShapedType type,
                           function_ref<InFlightDiagnostic()> emitError);

// This interface is used for HLO dialects that have accompanying
// BoundedAttrInterface attributes which can carry bounds for dimension sizes
// of accompanying shaped types.
class BoundedDialectInterface
    : public DialectInterface::Base<BoundedDialectInterface> {
 public:
  explicit BoundedDialectInterface(Dialect *dialect) : Base(dialect) {}
  virtual Attribute createBoundedAttr(ArrayRef<int64_t> bounds) const = 0;
};

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
      return isCompatibleForHloTypeInference(actual, expected);
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
      MLIRContext * /*context*/, Optional<Location> location,
      ValueRange operands, DictionaryAttr /*attributes*/,
      RegionRange /*regions*/, SmallVectorImpl<Type> &inferredReturnTypes) {
    // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
    // support quantization or sparsity.
    if (operands.empty())
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");

    if (failed(inferMostSpecificType(location, operands.getTypes(),
                                     inferredReturnTypes)))
      return failure();
    return success();
  }

  // This function is not going to be called automatically.
  // It needs to be paired with INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS
  // (see examples in StablehloOps.cc).
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
      Optional<Location> location, ValueTypeRange<ValueRange> inputTypes,
      SmallVectorImpl<Type> &inferredReturnTypes) {
    SmallVector<RankedTensorType> rankedTypes;
    for (auto inputType : inputTypes)
      if (auto rankedType = inputType.dyn_cast<RankedTensorType>())
        rankedTypes.push_back(rankedType);
    if (rankedTypes.empty()) {
      inferredReturnTypes.push_back(inputTypes[0]);
      return success();
    }

    auto rank = rankedTypes[0].getRank();
    BoundedDialectInterface *dialect = nullptr;
    SmallVector<int64_t> inferredDimSizes(rank, ShapedType::kDynamicSize);
    SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamicSize);
    for (auto rankedType : rankedTypes) {
      SmallVector<int64_t> bounds;
      if (auto boundedAttr = rankedType.getEncoding()
                                 .dyn_cast_or_null<BoundedAttrInterface>()) {
        dialect = cast<BoundedDialectInterface>(&boundedAttr.getDialect());
        bounds = llvm::to_vector<4>(boundedAttr.getBounds());
      } else if (rankedType.getEncoding()) {
        // TODO(zhouxin) infer sparsity encoding after b/238903065 is fixed.
        inferredReturnTypes.push_back(inputTypes[0]);
        return success();
      }

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
                     [](auto el) { return el != ShapedType::kDynamicSize; })) {
      encoding = dialect->createBoundedAttr(inferredBounds);
    }
    inferredReturnTypes.push_back(RankedTensorType::get(
        inferredDimSizes, rankedTypes[0].getElementType(), encoding));

    return success();
  }
};

}  // namespace OpTrait
}  // namespace hlo
}  // namespace mlir

#endif
