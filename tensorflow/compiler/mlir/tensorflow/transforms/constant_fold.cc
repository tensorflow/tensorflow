/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"

namespace mlir {
namespace TF {

// Implements a TF specific policy on when constant folding is allowed.
// Policy:
//
// Find the total size of the operands and results, ignoring types with
// undefined bit widths and unknown shapes.
// Disable constant folding if operands size is greater than a certain threshold
// (`kOperandsSizeThreshold`).
//
// Otherwise, allow folding if:
// 1. size of results is less than a certain threshold
//    (`kResultsSizeThreshold`), or
// 2. size of results is within a factor (`kSizeFactor`) of size of operands.
// TODO(b/157226221): Look into other heuristics for constant fold policy.
static bool IsFoldedByDefaultPolicy(Operation* inst) {
  auto get_size = [&](TypeRange types) {
    int64_t size = 0;
    for (auto t : types) {
      auto tensor_type = mlir::cast<TensorType>(t);
      // Ignore types with undefined bit widths.
      if (!tensor_type.getElementType().isIntOrFloat()) continue;
      // Ignore types with dynamic shapes.
      if (!tensor_type.hasStaticShape()) continue;
      size += tensor_type.getNumElements() *
              tensor_type.getElementType().getIntOrFloatBitWidth();
    }
    return size;
  };

  int64_t results_size = get_size(inst->getResultTypes());
  int64_t operands_size = get_size(inst->getOperandTypes());

  constexpr int kSizeFactor = 2;
  constexpr int64_t kResultsSizeThreshold = (1 << 16);   // 64 Kib =   8 KiB
  constexpr int64_t kOperandsSizeThreshold = (1 << 30);  //  1 Gib = 128 MiB

  return (operands_size <= kOperandsSizeThreshold) &&
         ((results_size <= kResultsSizeThreshold) ||
          (results_size <= kSizeFactor * operands_size));
}

LogicalResult ConstantFoldFallbackHook(
    Operation* inst, ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult>& results) {  // NOLINT
  if (!CanBeFolded(inst)) return failure();

  // Determine if we should attempt to fold this operation by considering the
  // size/size increase due to folding.
  if (!IsFoldedByDefaultPolicy(inst)) return failure();

  // If all the results are empty and has numerical element types, set results
  // to empty elements attribute. This is restricted to the numerical element
  // types as the DenseElementsAttr only supports numerical and string types.
  // TODO(hinsu): Handle ops that have one of the results empty for constant
  // propagation.
  bool has_empty_numerical_results =
      llvm::all_of(inst->getResultTypes(), [](Type ty) {
        ShapedType shaped_ty = mlir::cast<ShapedType>(ty);
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results &&
      // TODO(jpienaar): Remove this once some unmodeled op behavior is
      // addressed.
      inst->isRegistered()) {
    for (Type ty : inst->getResultTypes()) {
      auto shaped_ty = mlir::cast<ShapedType>(ty);
      results.push_back(
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>()));
    }
    return success();
  }

  // Returns directly if any of the operands is not an elements attributes.
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !mlir::isa<ElementsAttr>(attr);
      }))
    return failure();

  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(mlir::cast<ElementsAttr>(input));
  }

  SmallVector<Attribute> constants;
  LogicalResult status = EvaluateOperation(inst, inputs, constants);
  results.assign(constants.begin(), constants.end());
  return status;
}

static bool init_hooks = ([] () {
  TensorFlowDialect::RegisterConstantFoldHook(ConstantFoldFallbackHook);
}(), true);

}  // namespace TF
}  // namespace mlir
