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

// This file defines the op traits used in the MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_

#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TF {

// Verifies if 'ref_type' is a REF type corresponding to 'type'.
static inline LogicalResult VerifyRefTypeMatch(mlir::Type type,
                                               mlir::Type maybe_ref_type) {
  if (auto ref_type = maybe_ref_type.dyn_cast<mlir::TF::TensorFlowRefType>())
    return success(ref_type.RemoveRef().getKind() == type.getKind());
  return failure();
}

// This class provides verification for ops that are known to have the same
// result types and all operands are either of the same type as result or a REF
// type corresponding to the result type.
template <typename ConcreteType>
class OperandsSameAsResultsTypeOrRef
    : public TraitBase<ConcreteType, OperandsSameAsResultsTypeOrRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    LogicalResult shapeMatch = impl::verifySameOperandsAndResultShape(op);
    if (failed(shapeMatch)) return shapeMatch;

    auto type = getElementTypeOrSelf(op->getResult(0).getType());

    // Verify that the first result type is same as the rest of the results.
    // We skip the comparison against itself.
    for (auto resultType : llvm::drop_begin(op->getResultTypes(), 1)) {
      resultType = getElementTypeOrSelf(resultType);
      if (resultType != type)
        return op->emitOpError() << "requires the same type for all results";
    }

    for (auto opType : op->getOperandTypes()) {
      opType = getElementTypeOrSelf(opType);
      if (opType != type && failed(VerifyRefTypeMatch(type, opType))) {
        return op->emitError() << "requires all operands to be either same "
                                  "as or ref type of results";
      }
    }
    return success();
  }
};

}  // namespace TF
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
