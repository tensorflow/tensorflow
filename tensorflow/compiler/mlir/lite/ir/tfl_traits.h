/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Op traits of the TFLite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TFL {

namespace detail {

// Verifies that all operands and results have the same element type (or type,
// if not shaped) after resolving TF reference types.
inline LogicalResult VerifySameOperandsAndResultElementTypeResolveRef(
    Operation* op) {
  Type element_type;
  if (op->getNumResults() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getResult(0).getType());
  } else if (op->getNumOperands() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getOperand(0).getType());
  } else {
    return success();
  }
  for (Type result_type : op->getResultTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(result_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  for (Type operand_type : op->getOperandTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(operand_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  return success();
}

// Merges the static dimensions of two shaped types, preferring the static size
// when one of the dimensions is dynamic. Returns the ranked type if one of the
// types is unranked.
inline ShapedType MergeType(ShapedType a, ShapedType b) {
  if (!a.hasRank()) return b;
  if (!b.hasRank()) return a;
  int64_t rank = a.getRank();
  llvm::SmallVector<int64_t, 4> dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim0 = a.getDimSize(i);
    int64_t dim1 = b.getDimSize(i);
    dims[i] = (dim0 == ShapedType::kDynamic) ? dim1 : dim0;
  }
  return RankedTensorType::get(dims, a.getElementType());
}

}  // namespace detail

// Verifies that the op has the same operand and result types after resolving
// TF reference types, allowing compatible shapes (e.g. `tensor<*xf32>` and
// `tensor<4xf32>`). Infers the result type from the operands.
//
// This is the TFLite counterpart of the TF dialect trait with the same name,
// so that the TFLite dialect does not depend on the TF dialect ops.
template <typename ConcreteType>
class SameOperandsAndResultTypeResolveRef
    : public TraitBase<ConcreteType, SameOperandsAndResultTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (failed(impl::verifySameOperandsAndResultShape(op))) return failure();
    return detail::VerifySameOperandsAndResultElementTypeResolveRef(op);
  }

  static LogicalResult inferReturnTypeComponentsFromOperands(
      MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
      DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
    if (operands.empty()) {
      return emitOptionalError(
          location,
          "Expected non-empty operands for [CompatibleOperandsAndResultType]");
    }
    auto result_ty = llvm::dyn_cast_or_null<ShapedType>(operands[0].getType());
    if (!result_ty) {
      return emitOptionalError(location, "Expected shape type for operand 0");
    }
    for (auto [index, ty] :
         llvm::drop_begin(llvm::enumerate(operands.getTypes()), 1)) {
      auto shape_type = llvm::dyn_cast_or_null<ShapedType>(ty);
      if (!shape_type) {
        return emitOptionalError(location, "Expected shape type for operand ",
                                 index);
      }
      result_ty = detail::MergeType(shape_type, result_ty);
    }
    inferredReturnShapes.push_back(result_ty);
    return success();
  }
};

}  // namespace TFL
}  // namespace OpTrait
}  // namespace mlir

// Defines `Op::inferReturnTypeComponents` for an op with the
// `TFL_SameOperandsAndResultTypeResolveRef` trait, which infers the result type
// from the operands.
#define TFL_INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)            \
  LogicalResult Op::inferReturnTypeComponents(                        \
      MLIRContext* context, std::optional<Location> location,         \
      ValueShapeRange operands, DictionaryAttr attributes,            \
      PropertyRef properties, RegionRange regions,                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {  \
    return inferReturnTypeComponentsFromOperands(                     \
        context, location, operands, attributes, properties, regions, \
        inferredReturnShapes);                                        \
  }

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_
