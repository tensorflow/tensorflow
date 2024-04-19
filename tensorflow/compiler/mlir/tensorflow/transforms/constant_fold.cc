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

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"
#include "tensorflow/core/platform/mutex.h"

namespace mlir {
namespace TF {

// Implements a TF specific policy on when constant folding is allowed.
// Policy:
//
// Disable constant folding if operands size is greater than a certain
// threshold (`kOperandsSizeThreshold`).
//
// Otherwise, allow folding if we do not know the shape of an operand or
// result i.e., one of these values has non-static shape. If we know all the
// shapes, find the total size of the operands and results. Folding of the op is
// allowed if one of the following conditions are met:
// 1. size of results is less than a certain threshold
// (`kResultsSizeThreshold`), or
// 2. size of results is within a factor (`kSizeFactor`) of size of operands, or
// TODO(b/157226221): Look into other heuristics for constant fold policy.
static bool IsFoldedByDefaultPolicy(Operation* inst) {
  bool has_unknown_shape = false;
  auto get_size = [&](TypeRange types) {
    int64_t size = 0;
    for (auto t : types) {
      auto tensor_type = t.cast<TensorType>();
      // Ignore types with undefined bit widths.
      if (!tensor_type.getElementType().isIntOrFloat()) continue;
      if (!tensor_type.hasStaticShape()) {
        has_unknown_shape = true;
        return size;
      }
      size += tensor_type.getNumElements() *
              tensor_type.getElementType().getIntOrFloatBitWidth();
    }
    return size;
  };

  int64_t results_size = get_size(inst->getResultTypes());
  int64_t operands_size = get_size(inst->getOperandTypes());

  constexpr int kSizeFactor = 2;
// TODO(b/233827625): Remove TF_DISABLE_CONSTANT_FOLDING macro.
#ifdef TF_DISABLE_CONSTANT_FOLDING
  constexpr int64_t kResultsSizeThreshold = 0;
#else
  constexpr int64_t kResultsSizeThreshold = (1 << 23);  // 1 MB
#endif
  constexpr int64_t kOperandsSizeThreshold = (1 << 30);  // 128 MB

  return (operands_size <= kOperandsSizeThreshold) &&
         (has_unknown_shape || (results_size <= kResultsSizeThreshold) ||
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
        ShapedType shaped_ty = ty.cast<ShapedType>();
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results &&
      // TODO(jpienaar): Remove this once some unmodeled op behavior is
      // addressed.
      inst->isRegistered()) {
    for (Type ty : inst->getResultTypes()) {
      auto shaped_ty = ty.cast<ShapedType>();
      results.push_back(
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>()));
    }
    return success();
  }

  // Returns directly if any of the operands is not an elements attributes.
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !attr.isa<ElementsAttr>();
      }))
    return failure();

  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(input.cast<ElementsAttr>());
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
